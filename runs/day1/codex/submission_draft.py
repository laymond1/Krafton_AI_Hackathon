from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

INPUT_BITS = 6
OUTPUT_BITS = 12
PROMPT_LEN = INPUT_BITS * 2
FULL_SEQ_LEN = PROMPT_LEN + OUTPUT_BITS
VOCAB_SIZE = 2
DEFAULT_SEED = 7


def int_to_lsb_bits(value: int, width: int) -> List[int]:
    return [(value >> bit_index) & 1 for bit_index in range(width)]


def lsb_bits_to_int(bits: Sequence[int]) -> int:
    return sum(int(bit) << bit_index for bit_index, bit in enumerate(bits))


def encode_prompt(a: int, b: int) -> List[int]:
    return int_to_lsb_bits(a, INPUT_BITS) + int_to_lsb_bits(b, INPUT_BITS)


def encode_product(a: int, b: int) -> List[int]:
    return int_to_lsb_bits(a * b, OUTPUT_BITS)


def encode_example(a: int, b: int) -> List[int]:
    return encode_prompt(a, b) + encode_product(a, b)


def decode_product_bits(bits: Sequence[int]) -> int:
    if len(bits) != OUTPUT_BITS:
        raise ValueError(f"Expected {OUTPUT_BITS} bits, got {len(bits)}")
    return lsb_bits_to_int(bits)


def worked_examples() -> List[Tuple[int, int]]:
    return [(0, 0), (23, 37), (63, 63)]


def verify_worked_examples() -> List[str]:
    results: List[str] = []
    for a, b in worked_examples():
        prompt = encode_prompt(a, b)
        product = encode_product(a, b)
        sequence = encode_example(a, b)
        results.append(
            f"{a} x {b}: prompt={prompt} product={product} sequence={sequence}"
        )
    return results


def causal_mask(seq_len: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


def sinusoidal_positions(seq_len: int, d_model: int, device: torch.device) -> Tensor:
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    scale = -math.log(10000.0) / max(1, d_model // 2 - 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * scale)
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class MultiplicationDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[int, int]]):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tensor:
        a, b = self.pairs[index]
        return torch.tensor(encode_example(a, b), dtype=torch.long)


def all_pairs() -> List[Tuple[int, int]]:
    return [(a, b) for a in range(64) for b in range(64)]


def random_pairs(count: int, seed: int) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randrange(64), rng.randrange(64)) for _ in range(count)]


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 4
    n_heads: int = 1
    head_dim: int = 4
    n_layers: int = 1
    ffn_dim: int = 8
    rope_theta: float = 3.0
    embedding_style: str = "learned"
    attention_style: str = "separate"
    activation: str = "swiglu"
    tie_o_to_q: bool = False
    share_norm: bool = False
    tie_output_head: bool = True
    use_fixed_pe: bool = False
    use_output_pos_bias: bool = False
    dropout: float = 0.0


def unit_rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def rotate_pairs(x: Tensor) -> Tensor:
    if x.size(-1) % 2 != 0:
        raise ValueError(f"Expected even last dimension, got {x.size(-1)}")
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack([-odd, even], dim=-1)
    return rotated.flatten(start_dim=-2)


def apply_rope(x: Tensor, theta: float) -> Tensor:
    head_dim = x.size(-1)
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half_dim = head_dim // 2
    positions = torch.arange(x.size(-2), device=x.device, dtype=x.dtype)
    scales = torch.arange(half_dim, device=x.device, dtype=x.dtype) / max(1, half_dim)
    inv_freq = theta ** (-scales)
    angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
    left = x[..., :half_dim]
    right = x[..., half_dim:]
    return torch.cat([left * cos - right * sin, left * sin + right * cos], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.d_model))

    def forward(self, x: Tensor) -> Tensor:
        return unit_rms_norm(x) * self.weight


class LearnedBinaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(VOCAB_SIZE, config.d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def table(self) -> Tensor:
        return self.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]

    def as_linear(self, x: Tensor) -> Tensor:
        return x @ self.weight.t()


class QuadraticBinaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.base = nn.Parameter(torch.randn(1))
        self.curvature = nn.Parameter(torch.randn(1))

    def table(self) -> Tensor:
        d = torch.arange(VOCAB_SIZE, device=self.base.device, dtype=self.base.dtype)
        cols = [self.base - self.curvature * (d * d)]
        if self.d_model >= 2:
            cols.append(-d)
        if self.d_model >= 3:
            cols.append(2.0 * d - 1.0)
        while len(cols) < self.d_model:
            cols.append(torch.zeros_like(d))
        return torch.stack(cols[: self.d_model], dim=-1)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.table()[token_ids]

    def as_linear(self, x: Tensor) -> Tensor:
        return x @ self.table().t()


class ArcBinaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.radius = nn.Parameter(torch.ones(1))
        self.base_angle = nn.Parameter(torch.zeros(1))
        self.delta_angle = nn.Parameter(torch.ones(1))
        self.tail_scale = nn.Parameter(torch.ones(1))

    def table(self) -> Tensor:
        d = torch.arange(VOCAB_SIZE, device=self.radius.device, dtype=self.radius.dtype)
        angles = self.base_angle + self.delta_angle * d
        cols = [
            self.radius * torch.cos(angles),
            self.radius * torch.sin(angles),
        ]
        if self.d_model >= 3:
            cols.append(self.tail_scale * (2.0 * d - 1.0))
        while len(cols) < self.d_model:
            cols.append(torch.zeros_like(d))
        return torch.stack(cols[: self.d_model], dim=-1)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.table()[token_ids]

    def as_linear(self, x: Tensor) -> Tensor:
        return x @ self.table().t()


class TinyAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        inner_dim = config.n_heads * config.head_dim
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.kv_proj = None
        self.o_proj = None
        self.phase_angle = None
        self.phase_scale = None

        if config.attention_style == "phase_tied":
            self.phase_angle = nn.Parameter(torch.zeros(1))
            self.phase_scale = nn.Parameter(torch.ones(1))
            self.v_proj = nn.Linear(config.d_model, inner_dim, bias=False)
            self.o_proj = nn.Linear(inner_dim, config.d_model, bias=False)
        elif config.attention_style == "k_eq_v":
            self.q_proj = nn.Linear(config.d_model, inner_dim, bias=False)
            self.kv_proj = nn.Linear(config.d_model, inner_dim, bias=False)
            if not config.tie_o_to_q:
                self.o_proj = nn.Linear(inner_dim, config.d_model, bias=False)
        elif config.attention_style == "k_rot_q":
            self.q_proj = nn.Linear(config.d_model, inner_dim, bias=False)
            if not config.tie_o_to_q:
                self.o_proj = nn.Linear(inner_dim, config.d_model, bias=False)
        else:
            self.q_proj = nn.Linear(config.d_model, inner_dim, bias=False)
            self.k_proj = nn.Linear(config.d_model, inner_dim, bias=False)
            self.v_proj = nn.Linear(config.d_model, inner_dim, bias=False)
            if not config.tie_o_to_q:
                self.o_proj = nn.Linear(inner_dim, config.d_model, bias=False)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        inner_dim = self.config.n_heads * self.config.head_dim
        if self.config.attention_style == "phase_tied":
            x0 = x[..., 0]
            q_vec = torch.stack(
                [
                    self.phase_scale[0] * x0 * torch.cos(self.phase_angle[0]),
                    -self.phase_scale[0] * x0 * torch.sin(self.phase_angle[0]),
                ],
                dim=-1,
            )
            k_vec = torch.stack([x0, torch.zeros_like(x0)], dim=-1)
            v_vec = self.v_proj(x)
        elif self.config.attention_style == "k_eq_v":
            q_vec = self.q_proj(x)
            kv_vec = self.kv_proj(x)
            k_vec = kv_vec
            v_vec = kv_vec
        elif self.config.attention_style == "k_rot_q":
            q_vec = self.q_proj(x)
            k_vec = rotate_pairs(q_vec)
            v_vec = q_vec
        else:
            q_vec = self.q_proj(x)
            k_vec = self.k_proj(x)
            v_vec = self.v_proj(x)

        q = q_vec.view(batch_size, seq_len, self.config.n_heads, self.config.head_dim)
        k = k_vec.view(batch_size, seq_len, self.config.n_heads, self.config.head_dim)
        v = v_vec.view(batch_size, seq_len, self.config.n_heads, self.config.head_dim)

        q = apply_rope(unit_rms_norm(q.transpose(1, 2)), self.config.rope_theta)
        k = apply_rope(unit_rms_norm(k.transpose(1, 2)), self.config.rope_theta)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.config.head_dim)
        scores = scores.masked_fill(attn_mask.view(1, 1, seq_len, seq_len), -1e9)
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, inner_dim)
        if self.config.tie_o_to_q and self.q_proj is not None:
            return F.linear(attended, self.q_proj.weight.t())
        return self.o_proj(attended)


class TinyMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.config.activation == "relu2":
            gate = F.relu(self.gate_proj(x))
            up = F.relu(self.up_proj(x))
            return self.down_proj(gate * up)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        shared_norm = RMSNorm(config) if config.share_norm else None
        self.attn_norm = shared_norm if shared_norm is not None else RMSNorm(config)
        self.self_attn = TinyAttention(config)
        self.ffn_norm = shared_norm if shared_norm is not None else RMSNorm(config)
        self.mlp = TinyMLP(config)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = x + self.self_attn(self.attn_norm(x), attn_mask)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class TinyMultiplierTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.embedding_style == "quadratic":
            self.token_embedding = QuadraticBinaryEmbedding(config)
        elif config.embedding_style == "arc":
            self.token_embedding = ArcBinaryEmbedding(config)
        else:
            self.token_embedding = LearnedBinaryEmbedding(config)
        self.blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = self.blocks[0].attn_norm if config.share_norm else RMSNorm(config)
        self.output_head = None
        if not config.tie_output_head:
            self.output_head = nn.Linear(config.d_model, VOCAB_SIZE, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))
        self.output_pos_bias = None
        if config.use_output_pos_bias:
            self.output_pos_bias = nn.Parameter(torch.zeros(OUTPUT_BITS, VOCAB_SIZE))

    def forward(self, token_ids: Tensor) -> Tensor:
        seq_len = token_ids.size(1)
        x = self.token_embedding(token_ids)
        if self.config.use_fixed_pe:
            x = x + sinusoidal_positions(seq_len, self.config.d_model, token_ids.device)
        attn_mask = causal_mask(seq_len, token_ids.device)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)
        if self.output_head is not None:
            logits = self.output_head(x) + self.output_bias
        else:
            logits = self.token_embedding.as_linear(x) + self.output_bias
        if self.output_pos_bias is not None and seq_len >= PROMPT_LEN:
            output_steps = min(seq_len - (PROMPT_LEN - 1), OUTPUT_BITS)
            logits[:, PROMPT_LEN - 1 : PROMPT_LEN - 1 + output_steps, :] += (
                self.output_pos_bias[:output_steps].unsqueeze(0)
            )
        return logits


def build_model() -> nn.Module:
    return TinyMultiplierTransformer(
        ModelConfig(
            d_model=3,
            n_heads=1,
            head_dim=4,
            n_layers=1,
            ffn_dim=2,
            rope_theta=3.0,
            embedding_style="arc",
            attention_style="k_rot_q",
            activation="swiglu",
            tie_o_to_q=True,
            share_norm=True,
        )
    )


def candidate_configs() -> List[ModelConfig]:
    return [
        ModelConfig(
            d_model=2,
            n_heads=1,
            head_dim=2,
            n_layers=1,
            ffn_dim=2,
            rope_theta=19.0,
            embedding_style="quadratic",
            attention_style="phase_tied",
            activation="relu2",
            share_norm=True,
        ),
        ModelConfig(
            d_model=2,
            n_heads=1,
            head_dim=2,
            n_layers=1,
            ffn_dim=2,
            rope_theta=19.0,
            embedding_style="quadratic",
            attention_style="separate",
            activation="relu2",
            share_norm=True,
        ),
        ModelConfig(
            d_model=3,
            n_heads=1,
            head_dim=4,
            n_layers=1,
            ffn_dim=2,
            rope_theta=3.0,
            embedding_style="arc",
            attention_style="k_rot_q",
            activation="swiglu",
            tie_o_to_q=True,
            share_norm=True,
        ),
        ModelConfig(
            d_model=3,
            n_heads=1,
            head_dim=4,
            n_layers=1,
            ffn_dim=2,
            rope_theta=3.0,
            embedding_style="arc",
            attention_style="k_eq_v",
            activation="swiglu",
            tie_o_to_q=True,
            share_norm=True,
        ),
        ModelConfig(
            d_model=3,
            n_heads=1,
            head_dim=4,
            n_layers=1,
            ffn_dim=4,
            rope_theta=3.0,
            embedding_style="arc",
            attention_style="k_eq_v",
            activation="swiglu",
            tie_o_to_q=True,
            share_norm=True,
        ),
        ModelConfig(
            d_model=4,
            n_heads=1,
            head_dim=4,
            n_layers=1,
            ffn_dim=8,
            rope_theta=3.0,
            embedding_style="learned",
            attention_style="k_eq_v",
            activation="swiglu",
            tie_o_to_q=True,
            share_norm=True,
        ),
        ModelConfig(
            d_model=4,
            n_heads=1,
            head_dim=4,
            n_layers=1,
            ffn_dim=8,
            rope_theta=3.0,
            embedding_style="learned",
            attention_style="k_eq_v",
            activation="swiglu",
            tie_o_to_q=True,
            share_norm=False,
            tie_output_head=False,
        ),
        # candidate 8: control + output-position logit bias
        ModelConfig(
            d_model=4,
            n_heads=1,
            head_dim=4,
            n_layers=1,
            ffn_dim=8,
            rope_theta=3.0,
            embedding_style="learned",
            attention_style="k_eq_v",
            activation="swiglu",
            tie_o_to_q=True,
            share_norm=False,
            tie_output_head=False,
            use_output_pos_bias=True,
        ),
        # candidate 9: candidate 8 + 2-head routing + untied attention output
        ModelConfig(
            d_model=4,
            n_heads=2,
            head_dim=2,
            n_layers=1,
            ffn_dim=8,
            rope_theta=3.0,
            embedding_style="learned",
            attention_style="k_eq_v",
            activation="swiglu",
            tie_o_to_q=False,
            share_norm=False,
            tie_output_head=False,
            use_output_pos_bias=True,
        ),
        # candidate 10: candidate 9 + fully separate K/V/O for extra positive-bit freedom
        ModelConfig(
            d_model=4,
            n_heads=2,
            head_dim=2,
            n_layers=1,
            ffn_dim=8,
            rope_theta=3.0,
            embedding_style="learned",
            attention_style="separate",
            activation="swiglu",
            tie_o_to_q=False,
            share_norm=False,
            tie_output_head=False,
            use_output_pos_bias=True,
        ),
        ModelConfig(
            d_model=4,
            n_heads=1,
            head_dim=4,
            n_layers=1,
            ffn_dim=8,
            rope_theta=3.0,
            embedding_style="learned",
            attention_style="k_eq_v",
            activation="swiglu",
            tie_o_to_q=True,
            share_norm=False,
            tie_output_head=False,
            use_fixed_pe=True,
        ),
        # candidate 12: d=16, 2L, 2h — medium baseline
        ModelConfig(
            d_model=16,
            n_heads=2,
            head_dim=8,
            n_layers=2,
            ffn_dim=32,
            rope_theta=3.0,
            embedding_style="learned",
            attention_style="separate",
            activation="swiglu",
            tie_output_head=True,
        ),
        # candidate 13: d=32, 2L, 2h — conservative safe baseline
        ModelConfig(
            d_model=32,
            n_heads=2,
            head_dim=16,
            n_layers=2,
            ffn_dim=64,
            rope_theta=3.0,
            embedding_style="learned",
            attention_style="separate",
            activation="swiglu",
            tie_output_head=True,
        ),
        # candidate 14: d=64, 2L, 4h — safe ceiling
        ModelConfig(
            d_model=64,
            n_heads=4,
            head_dim=16,
            n_layers=2,
            ffn_dim=128,
            rope_theta=3.0,
            embedding_style="learned",
            attention_style="separate",
            activation="swiglu",
            tie_output_head=True,
        ),
    ]


def unique_parameter_count(module: nn.Module) -> int:
    seen = set()
    total = 0
    for parameter in module.parameters():
        pointer = parameter.data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        total += parameter.numel()
    return total


def output_loss(logits: Tensor, full_sequence: Tensor) -> Tensor:
    targets = full_sequence[:, 1:]
    output_logits = logits[:, PROMPT_LEN - 1 :, :]
    output_targets = targets[:, PROMPT_LEN - 1 :]
    return F.cross_entropy(
        output_logits.reshape(-1, VOCAB_SIZE),
        output_targets.reshape(-1),
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for full_sequence in dataloader:
        full_sequence = full_sequence.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(full_sequence[:, :-1])
        loss = output_loss(logits, full_sequence)
        loss.backward()
        optimizer.step()
        batch_size = full_sequence.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(1, total_items)


@torch.no_grad()
def greedy_decode_product(
    model: nn.Module, a: int, b: int, device: torch.device
) -> List[int]:
    model.eval()
    context = torch.tensor([encode_prompt(a, b)], dtype=torch.long, device=device)
    for _ in range(OUTPUT_BITS):
        logits = model(context)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        context = torch.cat([context, next_token], dim=1)
    return context[0, PROMPT_LEN:].tolist()


@torch.no_grad()
def exact_match_accuracy(
    model: nn.Module,
    pairs: Iterable[Tuple[int, int]],
    device: torch.device,
) -> float:
    total = 0
    correct = 0
    for a, b in pairs:
        prediction = greedy_decode_product(model, a, b, device)
        if prediction == encode_product(a, b):
            correct += 1
        total += 1
    return correct / max(1, total)


@torch.no_grad()
def bitwise_accuracy(
    model: nn.Module,
    pairs: Iterable[Tuple[int, int]],
    device: torch.device,
) -> float:
    total_bits = 0
    correct_bits = 0
    for a, b in pairs:
        prediction = greedy_decode_product(model, a, b, device)
        target = encode_product(a, b)
        correct_bits += sum(int(pred == gold) for pred, gold in zip(prediction, target))
        total_bits += OUTPUT_BITS
    return correct_bits / max(1, total_bits)


@torch.no_grad()
def per_bit_profile(
    model: nn.Module,
    pairs: Iterable[Tuple[int, int]],
    device: torch.device,
) -> dict:
    stats = [
        {
            "correct": 0,
            "total": 0,
            "false_positive": 0,
            "false_negative": 0,
            "pred_ones": 0,
            "target_ones": 0,
        }
        for _ in range(OUTPUT_BITS)
    ]
    first_wrong = [0 for _ in range(OUTPUT_BITS)]
    exact_matches = 0
    total_cases = 0

    for a, b in pairs:
        prediction = greedy_decode_product(model, a, b, device)
        target = encode_product(a, b)
        wrong_positions = []
        for bit_index, (pred_bit, target_bit) in enumerate(zip(prediction, target)):
            bit_stats = stats[bit_index]
            bit_stats["total"] += 1
            bit_stats["pred_ones"] += int(pred_bit == 1)
            bit_stats["target_ones"] += int(target_bit == 1)
            if pred_bit == target_bit:
                bit_stats["correct"] += 1
            else:
                wrong_positions.append(bit_index)
                if pred_bit == 1:
                    bit_stats["false_positive"] += 1
                else:
                    bit_stats["false_negative"] += 1
        if wrong_positions:
            first_wrong[wrong_positions[0]] += 1
        else:
            exact_matches += 1
        total_cases += 1

    return {
        "total_cases": total_cases,
        "exact_matches": exact_matches,
        "first_wrong": first_wrong,
        "bits": [
            {
                "bit": bit_index,
                "accuracy": bit_stats["correct"] / max(1, bit_stats["total"]),
                "false_positive_rate": bit_stats["false_positive"] / max(1, bit_stats["total"]),
                "false_negative_rate": bit_stats["false_negative"] / max(1, bit_stats["total"]),
                "pred_one_rate": bit_stats["pred_ones"] / max(1, bit_stats["total"]),
                "target_one_rate": bit_stats["target_ones"] / max(1, bit_stats["total"]),
            }
            for bit_index, bit_stats in enumerate(stats)
        ],
    }


@torch.no_grad()
def teacher_forced_bit_profile(
    model: nn.Module,
    pairs: Iterable[Tuple[int, int]],
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    sequences = [encode_example(a, b) for a, b in pairs]
    stats = [
        {
            "correct": 0,
            "total": 0,
            "false_positive": 0,
            "false_negative": 0,
            "pred_ones": 0,
            "target_ones": 0,
            "prob_one_sum": 0.0,
        }
        for _ in range(OUTPUT_BITS)
    ]
    first_wrong = [0 for _ in range(OUTPUT_BITS)]
    exact_matches = 0

    model.eval()
    for start in range(0, len(sequences), batch_size):
        batch = torch.tensor(
            sequences[start : start + batch_size], dtype=torch.long, device=device
        )
        logits = model(batch[:, :-1])[:, PROMPT_LEN - 1 :, :]
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        targets = batch[:, PROMPT_LEN:]
        wrong_mask = predictions.ne(targets)

        for bit_index in range(OUTPUT_BITS):
            bit_predictions = predictions[:, bit_index]
            bit_targets = targets[:, bit_index]
            bit_wrong = wrong_mask[:, bit_index]
            bit_stats = stats[bit_index]
            bit_stats["total"] += int(bit_targets.numel())
            bit_stats["correct"] += int((~bit_wrong).sum().item())
            bit_stats["false_positive"] += int(
                ((bit_predictions == 1) & (bit_targets == 0)).sum().item()
            )
            bit_stats["false_negative"] += int(
                ((bit_predictions == 0) & (bit_targets == 1)).sum().item()
            )
            bit_stats["pred_ones"] += int((bit_predictions == 1).sum().item())
            bit_stats["target_ones"] += int((bit_targets == 1).sum().item())
            bit_stats["prob_one_sum"] += float(probs[:, bit_index, 1].sum().item())

        first_wrong_batch = wrong_mask.float().argmax(dim=-1)
        wrong_rows = wrong_mask.any(dim=-1)
        for row_index, has_wrong in enumerate(wrong_rows.tolist()):
            if has_wrong:
                first_wrong[int(first_wrong_batch[row_index].item())] += 1
            else:
                exact_matches += 1

    return {
        "total_cases": len(sequences),
        "exact_matches": exact_matches,
        "first_wrong": first_wrong,
        "bits": [
            {
                "bit": bit_index,
                "accuracy": bit_stats["correct"] / max(1, bit_stats["total"]),
                "false_positive_rate": bit_stats["false_positive"] / max(1, bit_stats["total"]),
                "false_negative_rate": bit_stats["false_negative"] / max(1, bit_stats["total"]),
                "pred_one_rate": bit_stats["pred_ones"] / max(1, bit_stats["total"]),
                "target_one_rate": bit_stats["target_ones"] / max(1, bit_stats["total"]),
                "avg_prob_one": bit_stats["prob_one_sum"] / max(1, bit_stats["total"]),
            }
            for bit_index, bit_stats in enumerate(stats)
        ],
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_eval_pairs(eval_size: int, seed: int, eval_all_pairs: bool) -> List[Tuple[int, int]]:
    if eval_all_pairs:
        return all_pairs()
    return random_pairs(eval_size, seed + 1)


def train_model(
    config: ModelConfig,
    train_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[nn.Module, torch.device, DataLoader]:
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device.type}")
    model = TinyMultiplierTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    train_loader = DataLoader(
        MultiplicationDataset(random_pairs(train_size, seed)),
        batch_size=batch_size,
        shuffle=True,
    )
    return model, device, (optimizer, scheduler, train_loader)


def run_demo_train(
    config: ModelConfig,
    train_size: int,
    eval_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
    eval_all_pairs: bool = False,
) -> None:
    model, device, train_parts = train_model(
        config=config,
        train_size=train_size,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    optimizer, scheduler, train_loader = train_parts
    eval_pairs = build_eval_pairs(eval_size, seed, eval_all_pairs)

    print(f"config={config}")
    print(f"parameters={unique_parameter_count(model)}")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        accuracy = exact_match_accuracy(model, eval_pairs, device)
        bit_accuracy = bitwise_accuracy(model, eval_pairs, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch} lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} exact_match={accuracy:.4f} "
            f"bit_accuracy={bit_accuracy:.4f}"
        )
        scheduler.step()


def print_bit_report(report: dict, label: str) -> None:
    print(f"{label}_bit_profile")
    for bit in report["bits"]:
        avg_prob_one = bit.get("avg_prob_one")
        prob_suffix = "" if avg_prob_one is None else f" prob1={avg_prob_one:.4f}"
        print(
            f"P{bit['bit']:02d} acc={bit['accuracy']:.4f} "
            f"fp={bit['false_positive_rate']:.4f} "
            f"fn={bit['false_negative_rate']:.4f} "
            f"pred1={bit['pred_one_rate']:.4f} "
            f"gold1={bit['target_one_rate']:.4f}"
            f"{prob_suffix}"
        )
    print(f"{label}_first_wrong_bit")
    for bit_index, count in enumerate(report["first_wrong"]):
        print(f"P{bit_index:02d} first_wrong_count={count}")


def run_bit_analysis(
    config: ModelConfig,
    train_size: int,
    eval_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
    eval_all_pairs: bool = False,
) -> None:
    model, device, train_parts = train_model(
        config=config,
        train_size=train_size,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    optimizer, scheduler, train_loader = train_parts
    epoch_eval_pairs = random_pairs(eval_size, seed + 1)
    final_eval_pairs = build_eval_pairs(eval_size, seed, eval_all_pairs)

    print(f"config={config}")
    print(f"parameters={unique_parameter_count(model)}")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        accuracy = exact_match_accuracy(model, epoch_eval_pairs, device)
        bit_accuracy = bitwise_accuracy(model, epoch_eval_pairs, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch} lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} exact_match={accuracy:.4f} "
            f"bit_accuracy={bit_accuracy:.4f}"
        )
        scheduler.step()

    report = per_bit_profile(model, final_eval_pairs, device)
    teacher_forced_report = teacher_forced_bit_profile(
        model, final_eval_pairs, device, batch_size=batch_size
    )
    final_exact_match = report["exact_matches"] / max(1, report["total_cases"])
    final_bit_accuracy = sum(bit["accuracy"] for bit in report["bits"]) / OUTPUT_BITS
    teacher_forced_exact_match = teacher_forced_report["exact_matches"] / max(
        1, teacher_forced_report["total_cases"]
    )
    teacher_forced_bit_accuracy = (
        sum(bit["accuracy"] for bit in teacher_forced_report["bits"]) / OUTPUT_BITS
    )
    print(
        f"final_eval exact_match={final_exact_match:.4f} "
        f"bit_accuracy={final_bit_accuracy:.4f} total_cases={report['total_cases']}"
    )
    print(
        f"teacher_forced_eval exact_match={teacher_forced_exact_match:.4f} "
        f"bit_accuracy={teacher_forced_bit_accuracy:.4f} "
        f"total_cases={teacher_forced_report['total_cases']}"
    )
    print_bit_report(report, label="greedy")
    print_bit_report(teacher_forced_report, label="teacher_forced")


def get_config(candidate_index: int | None) -> ModelConfig:
    if candidate_index is None:
        return ModelConfig()
    configs = candidate_configs()
    if candidate_index < 0 or candidate_index >= len(configs):
        raise ValueError(
            f"candidate_index must be in [0, {len(configs) - 1}], got {candidate_index}"
        )
    return configs[candidate_index]


def run_candidate_sweep(
    epochs: int,
    train_size: int,
    eval_size: int,
    batch_size: int,
    seed: int,
    eval_all_pairs: bool = False,
) -> None:
    for index, config in enumerate(candidate_configs()):
        print(f"candidate_index={index}")
        run_demo_train(
            config=config,
            train_size=train_size,
            eval_size=eval_size,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + index,
            eval_all_pairs=eval_all_pairs,
        )


def smoke_test() -> None:
    set_seed(DEFAULT_SEED)
    device = torch.device("cpu")
    model = build_model().to(device)
    sample = torch.tensor([encode_example(23, 37)], dtype=torch.long)
    logits = model(sample[:, :-1])
    assert logits.shape == (1, FULL_SEQ_LEN - 1, VOCAB_SIZE)
    decoded = decode_product_bits(encode_product(63, 63))
    assert decoded == 3969
    print("worked_examples")
    for line in verify_worked_examples():
        print(line)
    print(f"smoke_test_logits_shape={tuple(logits.shape)}")
    print(f"default_parameter_count={unique_parameter_count(model)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--demo-train", action="store_true")
    parser.add_argument("--analyze-bits", action="store_true")
    parser.add_argument("--sweep-candidates", action="store_true")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-all-pairs", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if args.sweep_candidates:
        run_candidate_sweep(
            epochs=args.epochs,
            train_size=args.train_size,
            eval_size=args.eval_size,
            batch_size=args.batch_size,
            seed=args.seed,
            eval_all_pairs=args.eval_all_pairs,
        )
        return
    if args.analyze_bits:
        run_bit_analysis(
            config=get_config(args.candidate_index),
            train_size=args.train_size,
            eval_size=args.eval_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            eval_all_pairs=args.eval_all_pairs,
        )
        return
    if args.demo_train:
        run_demo_train(
            config=get_config(args.candidate_index),
            train_size=args.train_size,
            eval_size=args.eval_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            eval_all_pairs=args.eval_all_pairs,
        )
        return
    print("Candidate configs:")
    for config in candidate_configs():
        print(config)


if __name__ == "__main__":
    main()
