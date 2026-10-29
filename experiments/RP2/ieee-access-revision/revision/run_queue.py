"""
Overnight training queue for revision experiments.
Runs models/strategies sequentially; skips completed folds (resume-safe).

Usage:
  python revision/run_queue.py --phase grid3a
  python revision/run_queue.py --phase all10_best
  python revision/run_queue.py --phase vit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from revision.control import wait_if_paused, write_control, read_control

PY = ROOT / "venv313" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

# Phase 3a: full imbalance grid on 3 models
GRID3A = []
for model in ["vgg19", "swin_tiny", "efficientnetb0"]:
    for strat in ["baseline", "weighted_ce", "focal", "oversample", "low_lr_warmup"]:
        GRID3A.append((model, strat))

# Phase 3b placeholder: filled after 3a picks best 2 strategies
ALL10_MODELS = [
    "resnet18", "resnet50", "densenet121", "vgg19", "efficientnetb0",
    "mobilenetv3", "convnext_tiny", "deit_tiny", "swin_tiny", "vit_b_16",
]

VIT_EXTRA = [
    ("mobilevit_s", "baseline"),
    ("mobilevit_s", "low_lr_warmup"),
    ("tiny_vit_5m", "baseline"),
    ("tiny_vit_5m", "low_lr_warmup"),
    ("deit_tiny", "low_lr_warmup"),
    ("swin_tiny", "low_lr_warmup"),
    ("vit_b_16", "low_lr_warmup"),
]


def run_one(model, strategy, folds="1-5"):
    cmd = [
        str(PY), "-u", str(ROOT / "revision" / "train_cv.py"),
        "--model", model, "--strategy", strategy, "--folds", folds,
    ]
    print("\n" + "=" * 70)
    print("QUEUE:", " ".join(cmd))
    print("=" * 70, flush=True)
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONUNBUFFERED"] = "1"
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        print(f"FAILED ({r.returncode}): {model} {strategy}", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["grid3a", "mech", "all10_best", "all10_wce", "all10_lowlr",
                             "all10_baseline", "vit", "baseline_top"])
    ap.add_argument("--best-strats", default="low_lr_warmup,baseline",
                    help="Comma-separated strategies for all10_best (from Phase 3a pick)")
    args = ap.parse_args()

    jobs = []
    if args.phase == "grid3a":
        jobs = GRID3A
    elif args.phase == "mech":
        # Fold 3 instrumented-style: baseline + LR variants on collapsed models
        jobs = [
            ("vgg19", "baseline"),
            ("vgg19", "low_lr_warmup"),
            ("swin_tiny", "baseline"),
            ("swin_tiny", "low_lr_warmup"),
        ]
    elif args.phase == "all10_best":
        strats = [s.strip() for s in args.best_strats.split(",") if s.strip()]
        jobs = [(m, s) for s in strats for m in ALL10_MODELS]
    elif args.phase == "all10_wce":
        jobs = [(m, "weighted_ce") for m in ALL10_MODELS]
    elif args.phase == "all10_lowlr":
        jobs = [(m, "low_lr_warmup") for m in ALL10_MODELS]
    elif args.phase == "all10_baseline":
        jobs = [(m, "baseline") for m in ALL10_MODELS]
    elif args.phase == "vit":
        jobs = VIT_EXTRA
    elif args.phase == "baseline_top":
        jobs = [(m, "baseline") for m in
                ["efficientnetb0", "vgg19", "swin_tiny", "resnet18", "mobilenetv3"]]

    # Ensure control file exists in running state when queue starts
    ctrl = read_control()
    if ctrl.get("state") == "stop":
        write_control("running", "queue started (was stop)")
    elif ctrl.get("state") not in ("running", "paused"):
        write_control("running", "queue started")

    fails = 0
    for model, strat in jobs:
        state = wait_if_paused(label=f"queue before {model}/{strat}")
        if state == "stop":
            print(f"[STOP] queue halted before {model}/{strat}", flush=True)
            break
        code = run_one(model, strat)
        if code != 0:
            fails += 1
    print(f"\nQueue done. Failures: {fails}/{len(jobs)}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
