# IEEE Access revision (RP2)

This folder holds the IEEE Access revision / reviewer-response experiment code.

## Contents
- `revision/` — training queue, CV trainer, monitor, VinDr prep, analyses
- `reviewer-concern/` — reviewer concern source docs (if present)
- `revision_results_snapshot/` — small JSON/tables/figures only (checkpoints excluded)
- `start_training.bat` — resume helper

## Note
- Original remote experiment branches were **not** deleted.
- Large `revision_results/checkpoints/*.pt` and live training logs stay local / untracked to keep `main` lean.
- Active training may continue in the working clone; this is a snapshot for organization on `main`.
