"""Run the fixed training protocol (200 epochs, 100k pairs, AdamW, cosine LR)
for a specific candidate index. Evaluates every 20 epochs + final full eval."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from runs.day1.codex.submission_draft import (
    candidate_configs, TinyMultiplierTransformer, unique_parameter_count,
    random_pairs, all_pairs, MultiplicationDataset,
    train_epoch, exact_match_accuracy, bitwise_accuracy,
    set_seed,
)
import torch
from torch.utils.data import DataLoader

TRAIN_SIZE = 100_000
EPOCHS = 200
BATCH_SIZE = 256
SEED = 42
EVAL_SAMPLE = 512
EVAL_INTERVAL = 20


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(candidate_idx: int):
    configs = candidate_configs()
    config = configs[candidate_idx]
    set_seed(SEED)
    device = get_device()

    model = TinyMultiplierTransformer(config).to(device)
    params = unique_parameter_count(model)
    print(f"=== Candidate {candidate_idx} ===")
    print(f"Config: d={config.d_model} L={config.n_layers} h={config.n_heads} ff={config.ffn_dim}")
    print(f"Parameters: {params}")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_loader = DataLoader(
        MultiplicationDataset(random_pairs(TRAIN_SIZE, SEED)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    eval_pairs = random_pairs(EVAL_SAMPLE, SEED + 1)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        if epoch % EVAL_INTERVAL == 0 or epoch == 1 or epoch == EPOCHS:
            acc = exact_match_accuracy(model, eval_pairs, device)
            bacc = bitwise_accuracy(model, eval_pairs, device)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"epoch={epoch:3d} lr={lr:.6f} loss={loss:.4f} exact={acc:.4f} bit={bacc:.4f} t={elapsed:.0f}s")

    # Final full evaluation on all 4096 pairs
    print("\n=== Final full evaluation (all 4096 pairs) ===")
    full_pairs = all_pairs()
    final_acc = exact_match_accuracy(model, full_pairs, device)
    final_bacc = bitwise_accuracy(model, full_pairs, device)
    print(f"exact_match={final_acc:.4f} bit_accuracy={final_bacc:.4f}")
    print(f"P_2={params} Acc_2={final_acc:.4f}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 9  # default: d=32
    run(idx)
