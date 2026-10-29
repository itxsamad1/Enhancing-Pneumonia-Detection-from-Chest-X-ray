"""Summarize Phase 3a and recommend best 2 strategies for Phase 3b."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "revision_results" / "revision_kfold_results.json"

MODELS = ["vgg19", "swin_tiny", "efficientnetb0"]
STRATS = ["baseline", "weighted_ce", "focal", "oversample", "low_lr_warmup"]


def main():
    d = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = []
    print(f"{'model':<16} {'strategy':<16} {'acc':>7} {'f1':>7} {'rec':>7} "
          f"{'auc':>7} {'bal':>7} {'mcc':>7} {'col':>4}")
    for m in MODELS:
        for s in STRATS:
            sm = d[m]["strategies"][s].get("summary") or {}
            if not sm:
                folds = d[m]["strategies"][s]["folds"]
                keys = ["accuracy", "f1", "recall", "auc", "balanced_accuracy", "mcc"]
                sm = {f"{k}_mean": sum(folds[f][k] for f in folds) / len(folds) for k in keys}
                sm["n_collapsed"] = sum(1 for f in folds if folds[f].get("collapsed"))
            row = (
                m, s,
                sm.get("accuracy_mean", 0), sm.get("f1_mean", 0),
                sm.get("recall_mean", 0), sm.get("auc_mean", 0),
                sm.get("balanced_accuracy_mean", 0), sm.get("mcc_mean", 0),
                int(sm.get("n_collapsed", 0)),
            )
            rows.append(row)
            print(f"{m:<16} {s:<16} {row[2]:7.4f} {row[3]:7.4f} {row[4]:7.4f} "
                  f"{row[5]:7.4f} {row[6]:7.4f} {row[7]:7.4f} {row[8]:4d}")

    print("\n--- strategy averages across 3 models ---")
    agg = defaultdict(lambda: defaultdict(list))
    for m, s, acc, f1, rec, auc, bal, mcc, col in rows:
        agg[s]["f1"].append(f1)
        agg[s]["auc"].append(auc)
        agg[s]["bal"].append(bal)
        agg[s]["mcc"].append(mcc)
        agg[s]["rec"].append(rec)
        agg[s]["acc"].append(acc)
        agg[s]["col"].append(col)

    ranked = []
    for s, v in agg.items():
        mean = {k: sum(v[k]) / len(v[k]) for k in ("f1", "auc", "bal", "mcc", "rec", "acc")}
        col_total = sum(v["col"])
        # Prefer high F1/bal/MCC, low collapse. Penalize collapse heavily.
        score = mean["f1"] + 0.5 * mean["bal"] + 0.3 * mean["mcc"] - 0.05 * col_total
        ranked.append((score, s, mean, col_total))
        print(
            f"{s:<16} f1={mean['f1']:.4f} auc={mean['auc']:.4f} bal={mean['bal']:.4f} "
            f"mcc={mean['mcc']:.4f} rec={mean['rec']:.4f} col={col_total}/15  score={score:.4f}"
        )

    ranked.sort(reverse=True)
    best2 = [s for _, s, _, _ in ranked[:2]]
    print("\nRECOMMENDED_BEST2:", ",".join(best2))
    out = ROOT / "revision_results" / "tables" / "phase3a_strategy_pick.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "best2": best2,
        "ranked": [
            {"strategy": s, "score": round(sc, 4), "means": {k: round(v, 4) for k, v in mean.items()},
             "collapsed_folds": col}
            for sc, s, mean, col in ranked
        ],
    }, indent=2), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
