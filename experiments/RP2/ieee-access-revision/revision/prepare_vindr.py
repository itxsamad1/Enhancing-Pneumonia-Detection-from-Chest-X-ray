"""
Build VinDr-CXR external-validation manifests from the Kaggle VinBigData release.

Label mapping (stated in paper):
  positive = Consolidation | Infiltration | Lung Opacity (pneumonia-pattern opacity)
  negative = all radiologists agree "No finding"
  other findings only -> hard_negative set (reported separately)

Outputs:
  revision_results/vindr_manifest.csv
  revision_results/vindr_hard_negatives.csv
  revision_results/tables/vindr_prep_summary.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json
from collections import defaultdict

from revision.common import RESULTS_DIR, TABLE_DIR, ensure_dirs, save_json

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [
    ROOT / "raw_datasets" / "vindr_cxr" / "vinbigdata",
    ROOT / "raw_datasets" / "vindr_cxr",
]

PNEUMONIA_CLASSES = {"Consolidation", "Infiltration", "Lung Opacity"}
NO_FINDING = "No finding"


def find_data_root() -> Path:
    for c in CANDIDATES:
        train_csv = c / "train.csv"
        train_dir = c / "train"
        if train_csv.exists() and train_dir.exists():
            return c
    # search one level deeper
    base = ROOT / "raw_datasets" / "vindr_cxr"
    if base.exists():
        for p in base.rglob("train.csv"):
            if (p.parent / "train").exists():
                return p.parent
    raise FileNotFoundError(
        "VinDr data not found. Expected train.csv + train/ under raw_datasets/vindr_cxr/"
    )


def main():
    ensure_dirs()
    root = find_data_root()
    train_csv = root / "train.csv"
    train_dir = root / "train"
    print(f"Using VinDr root: {root}")

    # image_id -> list of (class_name, x_min, y_min, x_max, y_max, rad_id)
    anns = defaultdict(list)
    with open(train_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row["image_id"]
            cls = row["class_name"]
            # empty boxes for No finding
            def fnum(k):
                v = row.get(k, "")
                return float(v) if v not in ("", None) else None
            anns[iid].append({
                "class_name": cls,
                "x_min": fnum("x_min"),
                "y_min": fnum("y_min"),
                "x_max": fnum("x_max"),
                "y_max": fnum("y_max"),
                "rad_id": row.get("rad_id", ""),
            })

    main_rows = []
    hard_neg_rows = []
    pos_any = pos_maj = neg = hard = missing = 0

    for iid, rows in anns.items():
        img_path = train_dir / f"{iid}.png"
        if not img_path.exists():
            # some re-uploads use jpg
            img_path = train_dir / f"{iid}.jpg"
        if not img_path.exists():
            missing += 1
            continue

        classes = {r["class_name"] for r in rows}
        # radiologist-level pneumonia votes
        by_rad = defaultdict(set)
        for r in rows:
            by_rad[r["rad_id"]].add(r["class_name"])

        rad_pos = []
        for rad, clss in by_rad.items():
            rad_pos.append(bool(clss & PNEUMONIA_CLASSES))
        n_pos_votes = sum(rad_pos)
        n_rads = max(len(by_rad), 1)

        is_no_finding_only = classes == {NO_FINDING}
        has_pneumonia = bool(classes & PNEUMONIA_CLASSES)
        has_other = bool(classes - PNEUMONIA_CLASSES - {NO_FINDING})

        # collect pneumonia boxes (union of all rads)
        boxes = []
        for r in rows:
            if r["class_name"] in PNEUMONIA_CLASSES and r["x_min"] is not None:
                boxes.append([r["x_min"], r["y_min"], r["x_max"], r["y_max"]])

        if is_no_finding_only:
            label = 0
            neg += 1
            main_rows.append({
                "filepath": str(img_path),
                "image_id": iid,
                "label": label,
                "label_rule": "no_finding_all",
                "n_pos_votes": 0,
                "n_rads": n_rads,
                "boxes": json.dumps(boxes),
            })
        elif has_pneumonia:
            # any-radiologist positive
            label = 1
            pos_any += 1
            maj = 1 if n_pos_votes >= 2 else 0
            if maj:
                pos_maj += 1
            main_rows.append({
                "filepath": str(img_path),
                "image_id": iid,
                "label": label,
                "label_rule": "pneumonia_any",
                "majority_pneumonia": maj,
                "n_pos_votes": n_pos_votes,
                "n_rads": n_rads,
                "boxes": json.dumps(boxes),
            })
        elif has_other:
            hard += 1
            hard_neg_rows.append({
                "filepath": str(img_path),
                "image_id": iid,
                "label": 0,
                "label_rule": "other_findings_only",
                "classes": "|".join(sorted(classes)),
                "boxes": "[]",
            })

    out_main = RESULTS_DIR / "vindr_manifest.csv"
    out_hard = RESULTS_DIR / "vindr_hard_negatives.csv"
    fields = ["filepath", "image_id", "label", "label_rule", "majority_pneumonia",
              "n_pos_votes", "n_rads", "boxes"]
    with open(out_main, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in main_rows:
            if "majority_pneumonia" not in r:
                r["majority_pneumonia"] = 0
            w.writerow(r)
    with open(out_hard, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "image_id", "label", "label_rule",
                                          "classes", "boxes"])
        w.writeheader()
        w.writerows(hard_neg_rows)

    summary = {
        "root": str(root),
        "n_images_annotated": len(anns),
        "n_main": len(main_rows),
        "n_negative_no_finding": neg,
        "n_positive_any": pos_any,
        "n_positive_majority": pos_maj,
        "n_hard_negatives": hard,
        "n_missing_files": missing,
        "pneumonia_classes": sorted(PNEUMONIA_CLASSES),
        "manifest": str(out_main),
        "hard_negatives": str(out_hard),
    }
    save_json(summary, TABLE_DIR / "vindr_prep_summary.json")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_main}")
    print(f"Wrote {out_hard}")


if __name__ == "__main__":
    main()
