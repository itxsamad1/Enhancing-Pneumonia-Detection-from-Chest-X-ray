"""
Phase 1 (R1-5): Confusion matrices, per-fold collapse grid, McNemar b/c tables.
Uses saved predictions from kfold_results.json — no GPU required.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2 as chi2_dist

# Lightweight constants (avoid importing revision.common which needs torch)
ROOT = Path(__file__).resolve().parent.parent
ORIG_KFOLD = ROOT / "kfold_results.json"
FIG_DIR = ROOT / "revision_results" / "figures"
TABLE_DIR = ROOT / "revision_results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "vgg19": "VGG-19", "efficientnetb0": "EfficientNet-B0", "mobilenetv3": "MobileNetV3-Small",
    "convnext_tiny": "ConvNeXt-Tiny", "deit_tiny": "DeiT-Tiny", "swin_tiny": "Swin-Tiny",
    "vit_b_16": "ViT-B/16",
}

ORDER = [
    "resnet18", "resnet50", "densenet121", "vgg19", "efficientnetb0",
    "mobilenetv3", "convnext_tiny", "deit_tiny", "swin_tiny", "vit_b_16",
]


def is_collapsed(preds, labels, auc=None):
    preds = np.asarray(preds)
    if len(np.unique(preds)) == 1:
        return True
    if auc is not None and abs(float(auc) - 0.5) < 1e-6:
        return True
    labels = np.asarray(labels)
    pos = labels == 1
    if pos.sum() > 0 and (preds[pos] == 1).mean() < 0.01:
        return True
    return False


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def fold_cm(preds, labels):
    return confusion_matrix(labels, preds, labels=[0, 1])


def aggregate_model(data):
    preds_all, labels_all = [], []
    fold_cms = {}
    collapse = {}
    for fold_key, fd in sorted(data["folds"].items()):
        p = np.array(fd["preds"])
        l = np.array(fd["labels"])
        cm = fold_cm(p, l)
        fold_cms[fold_key] = cm.tolist()
        collapse[fold_key] = bool(is_collapsed(p, l, fd.get("auc")))
        preds_all.extend(p.tolist())
        labels_all.extend(l.tolist())
    agg = fold_cm(preds_all, labels_all)
    return {
        "display": data.get("display", ""),
        "aggregated_cm": agg.tolist(),
        "fold_cms": fold_cms,
        "collapse": collapse,
        "n_collapsed": int(sum(collapse.values())),
        "preds": preds_all,
        "labels": labels_all,
    }


def mcnemar_pair(p1, p2, labels):
    n = min(len(p1), len(p2), len(labels))
    p1, p2, lb = np.array(p1[:n]), np.array(p2[:n]), np.array(labels[:n])
    c1, c2 = (p1 == lb), (p2 == lb)
    b = int(np.sum(c1 & ~c2))
    c = int(np.sum(~c1 & c2))
    if b + c == 0:
        p = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = float(1 - chi2_dist.cdf(chi2, df=1))
    return {"b": b, "c": c, "p_value": round(p, 6), "significant": p < 0.05}


def plot_confusion_grid(results: dict, out: Path):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    axes = axes.ravel()
    for i, key in enumerate(ORDER):
        ax = axes[i]
        cm = np.array(results[key]["aggregated_cm"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
            xticklabels=["Normal", "Pneumonia"],
            yticklabels=["Normal", "Pneumonia"],
            annot_kws={"size": 10, "weight": "bold"},
        )
        n_col = results[key]["n_collapsed"]
        title = results[key]["display"]
        if n_col:
            title += f" *({n_col}/5 collapse)"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle(
        "Figure 21. Aggregated confusion matrices across 5 folds (all 10 architectures)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_collapse_heatmap(results: dict, out: Path):
    mat = np.zeros((len(ORDER), 5), dtype=int)
    for i, key in enumerate(ORDER):
        for j in range(5):
            fk = f"fold_{j+1}"
            mat[i, j] = 1 if results[key]["collapse"].get(fk, False) else 0
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(
        mat, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax,
        xticklabels=[f"Fold {i}" for i in range(1, 6)],
        yticklabels=[MODEL_REGISTRY[k] for k in ORDER],
        linewidths=0.5, linecolor="white",
    )
    ax.set_title(
        "Figure 22. Per-fold majority-class collapse (1 = collapsed, AUC≈0.50)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    ensure_dirs()
    with open(ORIG_KFOLD, encoding="utf-8") as f:
        raw = json.load(f)  # large file; preds kept per fold

    results = {}
    for key in ORDER:
        results[key] = aggregate_model(raw[key])
        print(
            f"{results[key]['display']:<20} collapsed={results[key]['n_collapsed']}/5  "
            f"CM={results[key]['aggregated_cm']}"
        )

    plot_confusion_grid(results, FIG_DIR / "Figure_21_confusion_matrices.png")
    plot_collapse_heatmap(results, FIG_DIR / "Figure_22_collapse_heatmap.png")

    # McNemar vs EfficientNet-B0 + the 3 non-significant pairs from the paper
    ref = "efficientnetb0"
    mcnemar = {}
    for key in ORDER:
        if key == ref:
            continue
        mcnemar[f"{ref}_vs_{key}"] = {
            "model_a": MODEL_REGISTRY[ref],
            "model_b": MODEL_REGISTRY[key],
            **mcnemar_pair(results[ref]["preds"], results[key]["preds"], results[ref]["labels"]),
        }

    special_pairs = [
        ("resnet18", "densenet121"),
        ("resnet18", "mobilenetv3"),
        ("densenet121", "mobilenetv3"),
    ]
    for a, b in special_pairs:
        mcnemar[f"{a}_vs_{b}"] = {
            "model_a": MODEL_REGISTRY[a],
            "model_b": MODEL_REGISTRY[b],
            **mcnemar_pair(results[a]["preds"], results[b]["preds"], results[a]["labels"]),
        }

    # Strip bulky preds from saved summary
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("preds", "labels")}
        for k, v in results.items()
    }
    save_json(summary, TABLE_DIR / "confusion_matrix_summary.json")
    save_json(mcnemar, TABLE_DIR / "mcnemar_bc_tables.json")

    # CSV-friendly table
    lines = ["Model,TN,FP,FN,TP,CollapsedFolds"]
    for key in ORDER:
        tn, fp, fn, tp = np.array(results[key]["aggregated_cm"]).ravel()
        lines.append(
            f"{MODEL_REGISTRY[key]},{tn},{fp},{fn},{tp},{results[key]['n_collapsed']}/5"
        )
    (TABLE_DIR / "confusion_matrix_table.csv").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved tables to {TABLE_DIR}")

    # Correction note for Swin-Tiny
    print("\n*** PAPER CORRECTION: Swin-Tiny collapsed in "
          f"{results['swin_tiny']['n_collapsed']}/5 folds "
          f"(folds {[k for k,v in results['swin_tiny']['collapse'].items() if v]}), "
          "not 3/5 as stated in the manuscript.")


if __name__ == "__main__":
    main()
