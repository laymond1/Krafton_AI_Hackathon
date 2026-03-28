from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn

import idea_eval_harness as harness
import submission_draft as core

STRUCTURE_DIM = 8


def build_structure_features() -> Tensor:
    features = torch.zeros(core.FULL_SEQ_LEN, STRUCTURE_DIM, dtype=torch.float32)

    for position in range(core.INPUT_BITS):
        features[position, 0] = 1.0
        features[position, 3] = position / max(1, core.INPUT_BITS - 1)

    for position in range(core.INPUT_BITS, core.PROMPT_LEN):
        local_pos = position - core.INPUT_BITS
        features[position, 1] = 1.0
        features[position, 4] = local_pos / max(1, core.INPUT_BITS - 1)

    for position in range(core.PROMPT_LEN, core.FULL_SEQ_LEN):
        local_pos = position - core.PROMPT_LEN
        features[position, 2] = 1.0
        features[position, 5] = local_pos / max(1, core.OUTPUT_BITS - 1)
        features[position, 6] = (local_pos - 5.5) / 5.5
        features[position, 7] = 1.0

    return features


class SharedRecurrentMultiplier(nn.Module):
    """Reuses one decoder block multiple times and optionally re-injects the input state."""

    def __init__(self, config: core.ModelConfig, recurrence_steps: int = 3):
        super().__init__()
        self.config = config
        self.recurrence_steps = recurrence_steps
        self.token_embedding = core.LearnedBinaryEmbedding(config)
        self.structure_proj = nn.Linear(STRUCTURE_DIM, config.d_model, bias=False)
        self.shared_block = core.DecoderBlock(config)
        self.final_norm = core.RMSNorm(config)
        self.input_injection_scale = nn.Parameter(torch.tensor(0.5))
        self.output_head = nn.Linear(config.d_model, core.VOCAB_SIZE, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(core.VOCAB_SIZE))
        self.output_pos_bias = nn.Parameter(torch.zeros(core.OUTPUT_BITS, core.VOCAB_SIZE))
        self.register_buffer("structure_features", build_structure_features(), persistent=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        seq_len = token_ids.size(1)
        base = self.token_embedding(token_ids)
        base = base + self.structure_proj(self.structure_features[:seq_len]).unsqueeze(0)
        if self.config.use_fixed_pe:
            base = base + core.sinusoidal_positions(seq_len, self.config.d_model, token_ids.device)

        x = base
        attn_mask = core.causal_mask(seq_len, token_ids.device)
        for _ in range(self.recurrence_steps):
            x = x + (self.input_injection_scale * base)
            x = self.shared_block(x, attn_mask)

        x = self.final_norm(x)
        logits = self.output_head(x) + self.output_bias
        if seq_len >= core.PROMPT_LEN:
            output_steps = min(seq_len - (core.PROMPT_LEN - 1), core.OUTPUT_BITS)
            logits[:, core.PROMPT_LEN - 1 : core.PROMPT_LEN - 1 + output_steps, :] += (
                self.output_pos_bias[:output_steps].unsqueeze(0)
            )
        return logits


def default_config() -> core.ModelConfig:
    return core.ModelConfig(
        d_model=12,
        n_heads=2,
        head_dim=6,
        n_layers=1,
        ffn_dim=24,
        rope_theta=3.0,
        embedding_style="learned",
        attention_style="separate",
        activation="swiglu",
        tie_o_to_q=False,
        share_norm=False,
        tie_output_head=False,
        use_fixed_pe=False,
        use_output_pos_bias=False,
    )


def smoke_test() -> None:
    device = torch.device("cpu")
    model = SharedRecurrentMultiplier(default_config()).to(device)
    sample = torch.tensor([core.encode_example(23, 37)], dtype=torch.long)
    logits = model(sample[:, :-1])
    assert logits.shape == (1, core.FULL_SEQ_LEN - 1, core.VOCAB_SIZE)
    print(f"smoke_test_logits_shape={tuple(logits.shape)}")
    print(f"parameter_count={core.unique_parameter_count(model)}")


def run_experiment(args: argparse.Namespace) -> None:
    config = default_config()
    model = SharedRecurrentMultiplier(config, recurrence_steps=args.recurrence_steps)
    eval_pairs = harness.build_eval_pairs(args.eval_all_pairs, args.eval_size, args.seed)
    model, device = harness.train_model_instance(
        model=model,
        train_size=args.train_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        eval_pairs=eval_pairs,
    )
    print(
        f"idea=shared_recurrent recurrence_steps={args.recurrence_steps} "
        f"config={config}"
    )
    harness.print_evaluation(harness.evaluate_model(model, eval_pairs, device), label="final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--recurrence-steps", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=core.DEFAULT_SEED)
    parser.add_argument("--eval-all-pairs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
