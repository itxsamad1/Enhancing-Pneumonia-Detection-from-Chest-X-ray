"""
Resumable 5-fold CV trainer for revision experiments.
Saves checkpoints, per-sample probs/paths, and metrics.

Usage:
  python revision/train_cv.py --model vgg19 --strategy baseline --folds 1-5
  python revision/train_cv.py --model efficientnetb0 --strategy weighted_ce
  python revision/train_cv.py --model swin_tiny --strategy low_lr_warmup --folds 3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import time
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
)

from revision.common import (
    MODEL_REGISTRY, ChestXrayDataset, TransformSubset, build_model,
    make_criterion, learning_rate_for, is_collapsed,
    train_transform, val_transform, save_json, load_json, ensure_dirs,
    N_FOLDS, MAX_EPOCHS, PATIENCE, BATCH_SIZE, WEIGHT_DECAY,
    GRADIENT_CLIP, NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS, PREFETCH_FACTOR,
    RANDOM_STATE, CKPT_DIR, RESULTS_DIR,
)
from revision.control import wait_if_paused, read_control, write_heartbeat


def parse_folds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def _loader_kwargs() -> dict:
    kw = {"num_workers": NUM_WORKERS, "pin_memory": PIN_MEMORY}
    if NUM_WORKERS > 0:
        kw["persistent_workers"] = PERSISTENT_WORKERS
        kw["prefetch_factor"] = PREFETCH_FACTOR
    return kw


def compute_metrics(labels, preds, probs):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = 0.5
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "specificity": float(spec),
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "collapsed": bool(is_collapsed(preds, labels, auc)),
    }


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        imgs, lbls = batch[0], batch[1]
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(imgs)
            loss = criterion(out, lbls)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == lbls).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_loader(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels, all_probs, all_paths = [], [], [], []
    for batch in loader:
        imgs, lbls, paths = batch[0], batch[1], batch[2]
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(imgs)
            loss = criterion(out, lbls)
        probs = torch.softmax(out.float(), dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(lbls.cpu().numpy().tolist())
        all_paths.extend(list(paths))
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["val_loss"] = total_loss / max(total, 1)
    metrics["preds"] = all_preds
    metrics["labels"] = all_labels
    metrics["probs"] = all_probs
    metrics["paths"] = all_paths
    return metrics


def run_fold(model_key, strategy, fold_idx, train_idx, val_idx, dataset, device, results):
    fold_key = f"fold_{fold_idx}"
    if fold_key in results.get(model_key, {}).get("strategies", {}).get(strategy, {}).get("folds", {}):
        m = results[model_key]["strategies"][strategy]["folds"][fold_key]
        print(f"  Fold {fold_idx} SKIPPED (done) Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
              f"collapsed={m.get('collapsed')}")
        return results

    print(f"\n  Fold {fold_idx}/{N_FOLDS}  train={len(train_idx)} val={len(val_idx)}  "
          f"strategy={strategy}", flush=True)
    write_heartbeat(
        model=model_key, strategy=strategy, fold=fold_idx, epoch=0,
        phase="fold_start", status=f"starting fold {fold_idx}",
    )

    train_sub = TransformSubset(dataset, train_idx, train_transform)
    val_sub = TransformSubset(dataset, val_idx, val_transform)

    train_labels = [dataset.labels[i] for i in train_idx]
    n0 = train_labels.count(0)
    n1 = train_labels.count(1)
    criterion, use_sampler = make_criterion(strategy, (n0, n1))
    if hasattr(criterion, "weight") and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device)

    lkw = _loader_kwargs()
    if use_sampler:
        counts = np.bincount(train_labels, minlength=2).astype(float)
        sample_w = [1.0 / counts[y] for y in train_labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(
            train_sub, batch_size=BATCH_SIZE, sampler=sampler,
            drop_last=True, **lkw,
        )
    else:
        train_loader = DataLoader(
            train_sub, batch_size=BATCH_SIZE, shuffle=True,
            drop_last=True, **lkw,
        )
    val_loader = DataLoader(
        val_sub, batch_size=BATCH_SIZE, shuffle=False, **lkw,
    )

    model = build_model(model_key).to(device)
    lr = learning_rate_for(strategy, model_key)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # warmup for low_lr_warmup: linear warmup over first 3 epochs
    warmup_epochs = 3 if strategy == "low_lr_warmup" else 0
    base_lr = lr

    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = None
    best_state = None
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # Honor dashboard Pause / Stop between epochs (safe; does not corrupt ckpts)
        state = wait_if_paused(
            label=f"{model_key}/{strategy} fold{fold_idx} before ep{epoch}"
        )
        if state == "stop":
            print(f"  [STOP] requested — ending fold {fold_idx} without saving this fold",
                  flush=True)
            return results

        write_heartbeat(
            model=model_key, strategy=strategy, fold=fold_idx, epoch=epoch,
            phase="training", status=f"running epoch {epoch}",
        )

        if warmup_epochs and epoch <= warmup_epochs:
            warm_lr = base_lr * epoch / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = warm_lr

        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        metrics = eval_loader(model, val_loader, criterion, device)
        scheduler.step(metrics["val_loss"])

        metrics_line = (
            f"    Ep {epoch:02d}  TrainLoss={t_loss:.4f} Acc={t_acc:.4f}  "
            f"ValLoss={metrics['val_loss']:.4f} Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f} "
            f"Rec={metrics['recall']:.4f} collapsed={metrics['collapsed']}"
        )
        print(metrics_line, flush=True)
        write_heartbeat(
            model=model_key, strategy=strategy, fold=fold_idx, epoch=epoch,
            phase="epoch_done", status="epoch finished",
            last_metrics=metrics_line.strip(),
            train_loss=t_loss, train_acc=t_acc,
            val_loss=metrics["val_loss"], accuracy=metrics["accuracy"],
            f1=metrics["f1"], auc=metrics["auc"], recall=metrics["recall"],
            collapsed=bool(metrics["collapsed"]),
        )

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            patience_counter = 0
            best_metrics = {k: v for k, v in metrics.items()}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}", flush=True)
                break

    elapsed = (time.time() - t0) / 60
    print(f"  Fold {fold_idx} done in {elapsed:.1f} min | "
          f"Acc={best_metrics['accuracy']:.4f} F1={best_metrics['f1']:.4f} "
          f"collapsed={best_metrics['collapsed']}", flush=True)

    ckpt_path = CKPT_DIR / f"{model_key}__{strategy}__fold{fold_idx}.pt"
    torch.save({
        "model_key": model_key,
        "strategy": strategy,
        "fold": fold_idx,
        "state_dict": best_state,
        "metrics": {k: v for k, v in best_metrics.items()
                    if k not in ("preds", "labels", "probs", "paths")},
    }, ckpt_path)

    # persist fold
    if model_key not in results:
        results[model_key] = {"display": MODEL_REGISTRY.get(model_key, model_key), "strategies": {}}
    if strategy not in results[model_key]["strategies"]:
        results[model_key]["strategies"][strategy] = {"folds": {}}
    results[model_key]["strategies"][strategy]["folds"][fold_key] = best_metrics

    out_file = RESULTS_DIR / "revision_kfold_results.json"
    save_json(results, out_file)

    del model, optimizer, scheduler, scaler, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def summarize_strategy(results, model_key, strategy):
    folds = results[model_key]["strategies"][strategy]["folds"]
    keys = ["accuracy", "precision", "recall", "f1", "specificity", "auc",
            "balanced_accuracy", "mcc"]
    summary = {}
    for mk in keys:
        vals = [folds[f"fold_{i}"][mk] for i in range(1, N_FOLDS + 1)
                if f"fold_{i}" in folds]
        if vals:
            summary[f"{mk}_mean"] = round(float(statistics.mean(vals)), 4)
            summary[f"{mk}_std"] = round(float(statistics.stdev(vals)), 4) if len(vals) > 1 else 0.0
    summary["n_collapsed"] = sum(
        1 for i in range(1, N_FOLDS + 1)
        if f"fold_{i}" in folds and folds[f"fold_{i}"].get("collapsed")
    )
    results[model_key]["strategies"][strategy]["summary"] = summary
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument(
        "--strategy", default="baseline",
        choices=["baseline", "weighted_ce", "focal", "oversample",
                 "feature_smote", "low_lr_warmup", "threshold"],
    )
    parser.add_argument("--folds", default="1-5")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"DataLoader: batch={BATCH_SIZE} workers={NUM_WORKERS}", flush=True)

    out_file = RESULTS_DIR / "revision_kfold_results.json"
    results = load_json(out_file) if out_file.exists() else {}

    dataset = ChestXrayDataset()
    print(f"Dataset: {len(dataset)} images "
          f"(N={dataset.labels.count(0)}, P={dataset.labels.count(1)})")

    labels_all = np.array(dataset.labels)
    indices = np.arange(len(dataset))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_splits = list(skf.split(indices, labels_all))

    for fold_idx in parse_folds(args.folds):
        state = wait_if_paused(label=f"{args.model}/{args.strategy} before fold{fold_idx}")
        if state == "stop":
            print("[STOP] exiting train_cv before next fold", flush=True)
            break
        # Re-load results so skip logic sees folds saved by parallel writes
        if out_file.exists():
            results = load_json(out_file)
        train_idx, val_idx = fold_splits[fold_idx - 1]
        results = run_fold(
            args.model, args.strategy, fold_idx,
            train_idx, val_idx, dataset, device, results,
        )

    summary = summarize_strategy(results, args.model, args.strategy)
    save_json(results, out_file)
    print(f"\nSummary {MODEL_REGISTRY[args.model]} / {args.strategy}:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Results -> {out_file}")


if __name__ == "__main__":
    main()
