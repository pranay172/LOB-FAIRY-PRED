import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold

from mamba2_lob import (
    ModelConfig,
    Mamba2LOBModel,
    build_engineered_features_torch,
    inverse_target_transform,
    target_transform,
    weighted_corr_loss,
)


def weighted_pearson_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, -6.0, 6.0)
    w = np.maximum(np.abs(y_true), 1e-8)

    sum_w = np.sum(w)
    if sum_w <= 0:
        return 0.0

    mu_t = np.sum(w * y_true) / sum_w
    mu_p = np.sum(w * y_pred) / sum_w

    dt = y_true - mu_t
    dp = y_pred - mu_p

    cov = np.sum(w * dt * dp) / sum_w
    var_t = np.sum(w * dt * dt) / sum_w
    var_p = np.sum(w * dp * dp) / sum_w

    if var_t <= 0 or var_p <= 0:
        return 0.0
    return float(cov / (math.sqrt(var_t) * math.sqrt(var_p)))


def load_sequences(parquet_path: str, max_sequences: int = 0) -> Dict[str, np.ndarray]:
    table = pq.read_table(parquet_path)
    columns = table.column_names

    feature_cols = columns[3:35]
    target_cols = columns[35:37]

    seq_ix = table.column("seq_ix").to_numpy()
    step_in_seq = table.column("step_in_seq").to_numpy()

    order = np.lexsort((step_in_seq, seq_ix))
    seq_ix = seq_ix[order]
    step_in_seq = step_in_seq[order]

    n_rows = seq_ix.shape[0]
    features = np.empty((n_rows, len(feature_cols)), dtype=np.float32)
    targets = np.empty((n_rows, len(target_cols)), dtype=np.float32)

    for ix, col in enumerate(feature_cols):
        features[:, ix] = table.column(col).to_numpy()[order].astype(np.float32)
    for ix, col in enumerate(target_cols):
        targets[:, ix] = table.column(col).to_numpy()[order].astype(np.float32)

    seq_ids, counts = np.unique(seq_ix, return_counts=True)
    if not np.all(counts == 1000):
        raise ValueError("Each sequence must have exactly 1000 rows.")

    n_seq = len(seq_ids)
    if n_rows != n_seq * 1000:
        raise ValueError("Unexpected dataset shape after sequence grouping.")

    raw = features.reshape((n_seq, 1000, len(feature_cols)))
    y = targets.reshape((n_seq, 1000, len(target_cols)))

    expected_steps = np.arange(1000, dtype=step_in_seq.dtype)
    actual_steps = step_in_seq.reshape((n_seq, 1000))
    if not np.all(actual_steps == expected_steps[None, :]):
        raise ValueError("step_in_seq is not ordered from 0..999 inside each sequence.")

    if max_sequences > 0:
        max_sequences = min(max_sequences, n_seq)
        seq_ids = seq_ids[:max_sequences]
        raw = raw[:max_sequences]
        y = y[:max_sequences]

    return {
        "seq_ids": seq_ids.astype(np.int64),
        "raw": raw,
        "targets": y,
    }


def sample_batch(
    raw: np.ndarray,
    targets: np.ndarray,
    seq_indices: np.ndarray,
    batch_size: int,
    window: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((batch_size, window, raw.shape[-1]), dtype=np.float32)
    steps = np.zeros((batch_size, window), dtype=np.int64)
    y = np.zeros((batch_size, targets.shape[-1]), dtype=np.float32)

    for i in range(batch_size):
        seq_ptr = int(seq_indices[rng.integers(0, len(seq_indices))])
        pred_step = int(rng.integers(99, 1000))
        start = max(0, pred_step - window + 1)

        segment = raw[seq_ptr, start : pred_step + 1]
        seg_steps = np.arange(start, pred_step + 1, dtype=np.int64)

        seg_len = segment.shape[0]
        pad_len = window - seg_len
        if pad_len > 0:
            x[i, :pad_len, :] = segment[0]
            steps[i, :pad_len] = 0

        x[i, pad_len:, :] = segment
        steps[i, pad_len:] = seg_steps
        y[i] = targets[seq_ptr, pred_step]

    return x, steps, y


@torch.no_grad()
def evaluate(
    model: Mamba2LOBModel,
    raw: np.ndarray,
    targets: np.ndarray,
    seq_indices: np.ndarray,
    window: int,
    batch_size: int,
    eval_steps: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, float]:
    model.eval()

    preds = []
    truth = []
    losses = []

    for _ in range(eval_steps):
        x_np, step_np, y_np = sample_batch(raw, targets, seq_indices, batch_size, window, rng)

        x = torch.from_numpy(x_np).to(device)
        steps = torch.from_numpy(step_np).to(device)
        y = torch.from_numpy(target_transform(y_np)).to(device)

        feats = build_engineered_features_torch(x, steps)
        pred, _ = model(feats)

        loss = weighted_corr_loss(pred, y)
        losses.append(float(loss.item()))

        pred_inv = inverse_target_transform(pred.cpu().numpy())
        preds.append(pred_inv)
        truth.append(y_np)

    y_true = np.concatenate(truth, axis=0)
    y_pred = np.concatenate(preds, axis=0)

    t0 = weighted_pearson_np(y_true[:, 0], y_pred[:, 0])
    t1 = weighted_pearson_np(y_true[:, 1], y_pred[:, 1])
    return {
        "loss": float(np.mean(losses)),
        "t0": t0,
        "t1": t1,
        "weighted_pearson": float((t0 + t1) / 2.0),
    }


def build_splits(seq_ids: np.ndarray, n_folds: int, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(seq_ids))

    if GroupKFold is None:
        rng = np.random.default_rng(123)
        rng.shuffle(indices)
        cut = int(0.85 * len(indices))
        return indices[:cut], indices[cut:]

    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(indices, groups=seq_ids))
    if fold >= len(splits):
        raise ValueError(f"fold={fold} out of range for n_folds={n_folds}")
    train_idx, valid_idx = splits[fold]
    return train_idx.astype(np.int64), valid_idx.astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mamba2-LOB model")
    parser.add_argument("--train-path", type=str, default="datasets/train.parquet")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--steps-per-epoch", type=int, default=2500)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="checkpoints/mamba2_lob")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_sequences(args.train_path, max_sequences=args.max_sequences)
    seq_ids = data["seq_ids"]
    raw = data["raw"]
    targets = data["targets"]

    train_idx, valid_idx = build_splits(seq_ids, n_folds=args.n_folds, fold=args.fold)
    print(f"Train sequences: {len(train_idx)} | Valid sequences: {len(valid_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ModelConfig(
        input_dim=69,
        d_model=args.d_model,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
    )
    model = Mamba2LOBModel(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * args.steps_per_epoch
    warmup = max(1, args.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    rng = np.random.default_rng(args.seed)
    global_step = 0
    best_score = -1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = []

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}")
        for _ in pbar:
            x_np, step_np, y_np = sample_batch(
                raw=raw,
                targets=targets,
                seq_indices=train_idx,
                batch_size=args.batch_size,
                window=args.window,
                rng=rng,
            )

            x = torch.from_numpy(x_np).to(device)
            steps = torch.from_numpy(step_np).to(device)
            y = torch.from_numpy(target_transform(y_np)).to(device)

            feats = build_engineered_features_torch(x, steps)
            pred, _ = model(feats)
            loss = weighted_corr_loss(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss.append(float(loss.item()))
            global_step += 1

            pbar.set_postfix(
                loss=f"{np.mean(running_loss[-50:]):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        valid_metrics = evaluate(
            model=model,
            raw=raw,
            targets=targets,
            seq_indices=valid_idx,
            window=args.window,
            batch_size=args.batch_size,
            eval_steps=args.eval_steps,
            device=device,
            rng=rng,
        )

        print(
            "Validation: "
            f"loss={valid_metrics['loss']:.4f} "
            f"t0={valid_metrics['t0']:.5f} "
            f"t1={valid_metrics['t1']:.5f} "
            f"score={valid_metrics['weighted_pearson']:.5f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "args": vars(args),
            "epoch": epoch,
            "global_step": global_step,
            "valid_metrics": valid_metrics,
        }

        latest_path = out_dir / "last.pt"
        torch.save(ckpt, latest_path)

        if valid_metrics["weighted_pearson"] > best_score:
            best_score = valid_metrics["weighted_pearson"]
            best_path = out_dir / "best.pt"
            torch.save(ckpt, best_path)
            print(f"Saved new best checkpoint to {best_path}")

    summary = {
        "best_valid_score": best_score,
        "checkpoint_dir": str(out_dir),
    }
    with (out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()
