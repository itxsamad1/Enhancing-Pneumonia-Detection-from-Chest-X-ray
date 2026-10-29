@echo off
REM Resume revision training after reboot (skips finished folds).
cd /d "%~dp0"
if exist "venv313\Scripts\python.exe" (
  set PY=venv313\Scripts\python.exe
) else (
  set PY=python
)
echo Setting control.json to running...
"%PY%" -c "from revision.control import write_control; write_control('running','started from bat')"
echo Starting queue (all10_best = low_lr_warmup + baseline x 10 models)...
start "revision-queue" /MIN "%PY%" -u revision\run_queue.py --phase all10_best --best-strats low_lr_warmup,baseline
echo Starting monitor on http://127.0.0.1:8765 ...
start "revision-monitor" "%PY%" -u revision\monitor_status.py --web --port 8765
echo Done. Open http://127.0.0.1:8765 for Pause/Continue.
