from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

import submission_draft as core

DENSE_THRESHOLD = 4


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def popcount6(value: int) -> int:
    return int(value).bit_count()


def multiplication_trace(a: int, b: int) -> dict:
    carry = 0
    partial_counts = []
    carry_outs = []
    output_bits = []

    for output_col in range(core.OUTPUT_BITS):
        partial = 0
        for a_index in range(core.INPUT_BITS):
            b_index = output_col - a_index
            if 0 <= b_index < core.INPUT_BITS:
                partial += ((a >> a_index) & 1) * ((b >> b_index) & 1)
        total = partial + carry
        output_bits.append(total & 1)
        carry = total >> 1
        partial_counts.append(partial)
        carry_outs.append(carry)

    longest_chain = 0
    running = 0
    for carry_out in carry_outs:
        if carry_out > 0:
            running += 1
            longest_chain = max(longest_chain, running)
        else:
            running = 0

    return {
        "partial_counts": partial_counts,
        "carry_outs": carry_outs,
        "output_bits": output_bits,
        "carry_columns": sum(int(carry_out > 0) for carry_out in carry_outs),
        "max_carry": max(carry_outs, default=0),
        "longest_chain": longest_chain,
    }


def carry_bucket(a: int, b: int) -> str:
    longest_chain = multiplication_trace(a, b)["longest_chain"]
    if longest_chain == 0:
        return "chain_0"
    if longest_chain == 1:
        return "chain_1"
    if longest_chain == 2:
        return "chain_2"
    if longest_chain == 3:
        return "chain_3"
    return "chain_4_plus"


def evaluate_model(
    model: nn.Module,
    pairs: Iterable[tuple[int, int]],
    device: torch.device,
) -> dict:
    pair_list = list(pairs)
    total_cases = len(pair_list)
    exact_matches = 0
    bit_correct = 0
    bit_total = 0
    per_column_wrong = [0 for _ in range(core.OUTPUT_BITS)]
    dense_dense_total = 0
    dense_dense_exact = 0
    carry_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "exact": 0})

    model.eval()
    for a, b in pair_list:
        prediction = core.greedy_decode_product(model, a, b, device)
        target = core.encode_product(a, b)
        is_exact = prediction == target
        exact_matches += int(is_exact)

        for index, (pred_bit, gold_bit) in enumerate(zip(prediction, target)):
            if pred_bit == gold_bit:
                bit_correct += 1
            else:
                per_column_wrong[index] += 1
            bit_total += 1

        if popcount6(a) >= DENSE_THRESHOLD and popcount6(b) >= DENSE_THRESHOLD:
            dense_dense_total += 1
            dense_dense_exact += int(is_exact)

        bucket = carry_bucket(a, b)
        carry_stats[bucket]["total"] += 1
        carry_stats[bucket]["exact"] += int(is_exact)

    per_column = []
    for index, wrong_count in enumerate(per_column_wrong):
        per_column.append(
            {
                "bit": index,
                "error_rate": wrong_count / max(1, total_cases),
                "accuracy": 1.0 - (wrong_count / max(1, total_cases)),
            }
        )

    carry_buckets = []
    for bucket in sorted(carry_stats.keys()):
        bucket_total = carry_stats[bucket]["total"]
        bucket_exact = carry_stats[bucket]["exact"]
        carry_buckets.append(
            {
                "bucket": bucket,
                "total": bucket_total,
                "exact_match": bucket_exact / max(1, bucket_total),
            }
        )

    return {
        "total_cases": total_cases,
        "exact_match": exact_matches / max(1, total_cases),
        "bit_accuracy": bit_correct / max(1, bit_total),
        "dense_dense_total": dense_dense_total,
        "dense_dense_exact_match": dense_dense_exact / max(1, dense_dense_total),
        "per_column": per_column,
        "carry_buckets": carry_buckets,
    }


def print_evaluation(report: dict, label: str) -> None:
    print(
        f"{label} exact_match={report['exact_match']:.4f} "
        f"bit_accuracy={report['bit_accuracy']:.4f} "
        f"dense_dense_exact_match={report['dense_dense_exact_match']:.4f} "
        f"dense_dense_total={report['dense_dense_total']}"
    )
    print(f"{label}_per_column")
    for column in report["per_column"]:
        print(
            f"P{column['bit']:02d} acc={column['accuracy']:.4f} "
            f"err={column['error_rate']:.4f}"
        )
    print(f"{label}_carry_buckets")
    for bucket in report["carry_buckets"]:
        print(
            f"{bucket['bucket']} total={bucket['total']} "
            f"exact_match={bucket['exact_match']:.4f}"
        )


def train_model_instance(
    model: nn.Module,
    train_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
    eval_pairs: Sequence[tuple[int, int]],
) -> tuple[nn.Module, torch.device]:
    core.set_seed(seed)
    device = detect_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    train_loader = DataLoader(
        core.MultiplicationDataset(core.random_pairs(train_size, seed)),
        batch_size=batch_size,
        shuffle=True,
    )

    print(f"device={device.type}")
    print(f"parameters={core.unique_parameter_count(model)}")
    for epoch in range(1, epochs + 1):
        train_loss = core.train_epoch(model, train_loader, optimizer, device)
        report = evaluate_model(model, eval_pairs, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"exact_match={report['exact_match']:.4f} "
            f"bit_accuracy={report['bit_accuracy']:.4f}"
        )
        scheduler.step()

    return model, device


def build_eval_pairs(eval_all_pairs: bool, eval_size: int, seed: int) -> list[tuple[int, int]]:
    if eval_all_pairs:
        return core.all_pairs()
    return core.random_pairs(eval_size, seed + 1)


def run_core_candidate(args: argparse.Namespace) -> None:
    config = core.get_config(args.candidate_index)
    model = core.TinyMultiplierTransformer(config)
    eval_pairs = build_eval_pairs(args.eval_all_pairs, args.eval_size, args.seed)
    model, device = train_model_instance(
        model=model,
        train_size=args.train_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        eval_pairs=eval_pairs,
    )
    print(f"config={config}")
    print_evaluation(evaluate_model(model, eval_pairs, device), label="final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=core.DEFAULT_SEED)
    parser.add_argument("--eval-all-pairs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_core_candidate(args)


if __name__ == "__main__":
    main()
