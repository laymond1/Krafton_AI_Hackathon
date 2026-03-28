from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runs.day1.codex.submission_draft import (
    OUTPUT_BITS,
    all_pairs,
    bitwise_accuracy,
    candidate_configs,
    exact_match_accuracy,
    random_pairs,
    teacher_forced_bit_profile,
    train_epoch,
    train_model,
    unique_parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-a", type=int, default=6)
    parser.add_argument("--candidate-b", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--greedy-eval-size", type=int, default=512)
    parser.add_argument("--save-report-dir")
    return parser.parse_args()


def run_candidate(
    index: int,
    epochs: int,
    train_size: int,
    batch_size: int,
    seed: int,
    greedy_eval_size: int,
) -> dict:
    config = candidate_configs()[index]
    model, device, train_parts = train_model(
        config=config,
        train_size=train_size,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    optimizer, scheduler, train_loader = train_parts
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
    return {
        "index": index,
        "config": config,
        "parameters": unique_parameter_count(model),
        "device": device,
        "model": model,
        "teacher_forced": teacher_forced_bit_profile(
            model, all_pairs(), device, batch_size=batch_size
        ),
        "greedy_pairs": random_pairs(greedy_eval_size, seed + 1000),
    }


def main() -> None:
    args = parse_args()
    results = []
    for offset, index in enumerate((args.candidate_a, args.candidate_b)):
        result = run_candidate(
            index=index,
            epochs=args.epochs,
            train_size=args.train_size,
            batch_size=args.batch_size,
            seed=args.seed + offset,
            greedy_eval_size=args.greedy_eval_size,
        )
        result["greedy_exact_match"] = exact_match_accuracy(
            result["model"], result["greedy_pairs"], result["device"]
        )
        result["greedy_bit_accuracy"] = bitwise_accuracy(
            result["model"], result["greedy_pairs"], result["device"]
        )
        results.append(result)

    for result in results:
        report = result["teacher_forced"]
        print(
            f"candidate={result['index']} parameters={result['parameters']} "
            f"device={result['device']} greedy_exact={result['greedy_exact_match']:.4f} "
            f"greedy_bit={result['greedy_bit_accuracy']:.4f}"
        )
        for bit in report["bits"]:
            if 2 <= bit["bit"] <= 8:
                print(
                    f"P{bit['bit']:02d} acc={bit['accuracy']:.4f} "
                    f"fn={bit['false_negative_rate']:.4f} "
                    f"pred1={bit['pred_one_rate']:.4f} "
                    f"gold1={bit['target_one_rate']:.4f} "
                    f"prob1={bit['avg_prob_one']:.4f}"
                )
        print()

    first, second = results
    print(
        f"delta candidate_{second['index']}_minus_{first['index']}"
    )
    delta_rows = []
    for bit_index in range(2, 9):
        bit_a = first["teacher_forced"]["bits"][bit_index]
        bit_b = second["teacher_forced"]["bits"][bit_index]
        row = {
            "bit": bit_index,
            "d_acc": bit_b["accuracy"] - bit_a["accuracy"],
            "d_fn": bit_b["false_negative_rate"] - bit_a["false_negative_rate"],
            "d_pred1": bit_b["pred_one_rate"] - bit_a["pred_one_rate"],
            "d_prob1": bit_b["avg_prob_one"] - bit_a["avg_prob_one"],
        }
        delta_rows.append(row)
        print(
            f"P{bit_index:02d} d_acc={row['d_acc']:+.4f} "
            f"d_fn={row['d_fn']:+.4f} "
            f"d_pred1={row['d_pred1']:+.4f} "
            f"d_prob1={row['d_prob1']:+.4f}"
        )

    if args.save_report_dir:
        out = Path(args.save_report_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary = {
            "candidate_a": results[0]["index"],
            "candidate_b": results[1]["index"],
            "epochs": args.epochs,
            "train_size": args.train_size,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "results": [
                {
                    "candidate_index": result["index"],
                    "parameters": result["parameters"],
                    "greedy_exact_match": result["greedy_exact_match"],
                    "greedy_bit_accuracy": result["greedy_bit_accuracy"],
                    "teacher_forced": result["teacher_forced"],
                }
                for result in results
            ],
            "delta_rows": delta_rows,
        }
        (out / "underprediction_compare.json").write_text(json.dumps(summary, indent=2))
        csv_lines = ["bit,d_acc,d_fn,d_pred1,d_prob1"]
        for row in delta_rows:
            csv_lines.append(
                f"{row['bit']},{row['d_acc']:.8f},{row['d_fn']:.8f},{row['d_pred1']:.8f},{row['d_prob1']:.8f}"
            )
        (out / "underprediction_delta.csv").write_text("\n".join(csv_lines) + "\n")


if __name__ == "__main__":
    main()
