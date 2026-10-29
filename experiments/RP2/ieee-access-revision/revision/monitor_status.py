"""
Training monitor + Pause / Continue / Start controls.

CLI (live refresh):
  python revision/monitor_status.py --watch

Web dashboard (auto-refresh every 10s):
  python revision/monitor_status.py --web
  then open http://127.0.0.1:8765

Buttons write revision_results/control.json; train_cv / run_queue honor pause.
Finished folds in revision_kfold_results.json are never re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "revision_results" / "revision_kfold_results.json"
CKPT_DIR = ROOT / "revision_results" / "checkpoints"
TERMINALS = Path(os.environ.get(
    "CURSOR_TERMINALS",
    r"C:\Users\Sam\.cursor\projects\d-Enhancing-Pneumonia-Detection-from-Chest-X-ray\terminals",
))

N_FOLDS = 5

ALL10_MODELS = [
    "resnet18", "resnet50", "densenet121", "vgg19", "efficientnetb0",
    "mobilenetv3", "convnext_tiny", "deit_tiny", "swin_tiny", "vit_b_16",
]
# Phase 3a pick: low_lr_warmup >> baseline (weighted_ce/focal collapsed on VGG/Swin)
BEST2 = ["low_lr_warmup", "baseline"]

# Active monitor queue — Phase 3b
ACTIVE_QUEUE = [(m, s) for s in BEST2 for m in ALL10_MODELS]
PHASE_NAME = "grid3b (all-10 × best-2: low_lr_warmup, baseline)"
TOTAL_JOBS = len(ACTIVE_QUEUE)  # 20
TOTAL_FOLDS = TOTAL_JOBS * N_FOLDS  # 100

DISPLAY = {
    "vgg19": "VGG-19", "swin_tiny": "Swin-Tiny", "efficientnetb0": "EfficientNet-B0",
    "resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "mobilenetv3": "MobileNetV3", "convnext_tiny": "ConvNeXt-Tiny",
    "deit_tiny": "DeiT-Tiny", "vit_b_16": "ViT-B/16",
}


def load_results() -> dict:
    if not RESULTS.exists():
        return {}
    try:
        return json.loads(RESULTS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def completed_folds(results: dict) -> list[tuple[str, str, str, dict]]:
    out = []
    for model, md in results.items():
        for strat, sd in md.get("strategies", {}).items():
            for fk, fd in sd.get("folds", {}).items():
                out.append((model, strat, fk, fd))
    return out


def find_training_log() -> Path | None:
    cands: list[tuple[float, Path]] = []
    # Detached queue log (preferred after reboot / Start button)
    qlog = ROOT / "revision_results" / "queue_stdout.log"
    if qlog.exists() and qlog.stat().st_size > 0:
        cands.append((qlog.stat().st_mtime, qlog))
    if TERMINALS.exists():
        for p in TERMINALS.glob("*.txt"):
            try:
                head = p.read_text(encoding="utf-8", errors="replace")[:4000]
            except Exception:
                continue
            if "run_queue.py" in head or "train_cv.py" in head:
                cands.append((p.stat().st_mtime, p))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def parse_live(log_path: Path | None) -> dict:
    live = {
        "log": str(log_path) if log_path else None,
        "queue_job": None,
        "model": None,
        "strategy": None,
        "fold": None,
        "epoch": None,
        "last_line": None,
        "last_metrics": None,
        "fold_done_mins": [],
    }
    if not log_path or not log_path.exists():
        return live
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return live

    # Keep last ~200k chars for speed
    text = text[-200_000:]
    lines = text.splitlines()

    for line in lines:
        m = re.search(r"--model\s+(\S+)\s+--strategy\s+(\S+)", line)
        if m:
            live["model"], live["strategy"] = m.group(1), m.group(2)
            live["queue_job"] = f"{m.group(1)} / {m.group(2)}"
        m = re.search(r"Fold\s+(\d+)/(\d+)", line)
        if m and "done" not in line.lower():
            live["fold"] = int(m.group(1))
        m = re.search(r"Ep\s+(\d+)", line)
        if m:
            live["epoch"] = int(m.group(1))
            live["last_metrics"] = line.strip()
        m = re.search(r"Fold\s+\d+\s+done in\s+([\d.]+)\s+min", line)
        if m:
            live["fold_done_mins"].append(float(m.group(1)))
        if line.strip():
            live["last_line"] = line.strip()[-200:]

    # Prefer last QUEUE line
    for line in reversed(lines):
        if "QUEUE:" in line and "--model" in line:
            m = re.search(r"--model\s+(\S+)\s+--strategy\s+(\S+)", line)
            if m:
                live["model"], live["strategy"] = m.group(1), m.group(2)
                live["queue_job"] = f"{m.group(1)} / {m.group(2)}"
            break
    return live


def fold_times_from_ckpts() -> list[float]:
    """Approximate fold durations from checkpoint mtimes (adjacent)."""
    if not CKPT_DIR.exists():
        return []
    files = sorted(CKPT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if len(files) < 2:
        return []
    mins = []
    for a, b in zip(files, files[1:]):
        dt = (b.stat().st_mtime - a.stat().st_mtime) / 60.0
        if 5 < dt < 180:  # sane fold duration
            mins.append(dt)
    return mins


def _merge_heartbeat(live: dict) -> dict:
    """Overlay live_heartbeat.json (freshest epoch) onto log-parsed live."""
    try:
        from revision.control import read_heartbeat
        hb = read_heartbeat()
    except Exception:
        hb = {}
    if not hb:
        return live
    # Ignore stale heartbeat (>2h) from a previous crashed run
    ts = hb.get("updated_ts")
    if ts and (time.time() - float(ts)) > 7200:
        return live
    if hb.get("model"):
        live["model"] = hb["model"]
    if hb.get("strategy"):
        live["strategy"] = hb["strategy"]
    if hb.get("fold") is not None:
        live["fold"] = hb["fold"]
    if hb.get("epoch") is not None and int(hb["epoch"]) > 0:
        live["epoch"] = int(hb["epoch"])
    if hb.get("last_metrics"):
        live["last_metrics"] = hb["last_metrics"]
    if hb.get("status"):
        live["status"] = hb["status"]
    live["heartbeat_at"] = hb.get("updated_at")
    return live


def build_status() -> dict:
    results = load_results()
    queue_set = set(ACTIVE_QUEUE)
    done_all = completed_folds(results)
    # Only count folds that belong to the active Phase 3b queue
    done = [(m, s, fk, fd) for m, s, fk, fd in done_all if (m, s) in queue_set]
    n_done = len(done)
    live = _merge_heartbeat(parse_live(find_training_log()))
    times = list(live["fold_done_mins"])
    if not times:
        times = [t for t in fold_times_from_ckpts() if 20 <= t <= 120]
    if len(times) >= 2:
        avg = sorted(times)[len(times) // 2]  # median
    elif times:
        avg = times[-1]
    else:
        avg = 55.0

    job_done = 0
    job_rows = []
    for model, strat in ACTIVE_QUEUE:
        folds = set()
        if model in results:
            folds = set(results[model].get("strategies", {}).get(strat, {}).get("folds", {}).keys())
        n = len(folds)
        if n >= N_FOLDS:
            job_done += 1
            state = "done"
        elif n > 0 or (live.get("model") == model and live.get("strategy") == strat):
            state = "running" if live.get("model") == model and live.get("strategy") == strat else "partial"
        else:
            state = "pending"
        job_rows.append({
            "model": model,
            "display": DISPLAY.get(model, model),
            "strategy": strat,
            "folds_done": n,
            "folds_total": N_FOLDS,
            "state": state,
            "collapsed": sum(
                1 for fk in folds
                if results[model]["strategies"][strat]["folds"][fk].get("collapsed")
            ) if n and model in results else 0,
        })

    remaining_folds = max(0, TOTAL_FOLDS - n_done)
    eta_min = remaining_folds * avg
    eta = str(timedelta(minutes=int(eta_min)))
    elapsed_note = None
    if CKPT_DIR.exists():
        pts = list(CKPT_DIR.glob("*.pt"))
        if pts:
            t0 = min(p.stat().st_mtime for p in pts)
            elapsed_note = str(timedelta(seconds=int(time.time() - t0)))

    pct = 100.0 * n_done / TOTAL_FOLDS if TOTAL_FOLDS else 0

    return {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": PHASE_NAME,
        "gpu_job_running": live.get("model") is not None,
        "current": {
            "architecture": DISPLAY.get(live.get("model") or "", live.get("model")),
            "model_key": live.get("model"),
            "strategy": live.get("strategy"),
            "fold": live.get("fold"),
            "epoch": live.get("epoch"),
            "status": live.get("status"),
            "last_metrics": live.get("last_metrics"),
            "last_line": live.get("last_line"),
            "heartbeat_at": live.get("heartbeat_at"),
        },
        "progress": {
            "folds_done": n_done,
            "folds_total": TOTAL_FOLDS,
            "jobs_done": job_done,
            "jobs_total": TOTAL_JOBS,
            "percent": round(pct, 1),
            "avg_min_per_fold": round(avg, 1),
            "eta": eta,
            "elapsed_since_first_ckpt": elapsed_note,
        },
        "recent_folds": [
            {
                "model": DISPLAY.get(m, m),
                "strategy": s,
                "fold": fk,
                "acc": round(fd.get("accuracy", 0), 4),
                "f1": round(fd.get("f1", 0), 4),
                "recall": round(fd.get("recall", 0), 4),
                "collapsed": bool(fd.get("collapsed")),
            }
            for m, s, fk, fd in done[-8:]
        ],
        "queue": job_rows,
        "log_file": live.get("log"),
        "control": _control_status(),
        "trainer_alive": _trainer_alive(),
    }


def _control_status() -> dict:
    try:
        from revision.control import read_control
        return read_control()
    except Exception:
        return {"state": "unknown", "message": "", "updated_at": None}


def _trainer_alive() -> bool:
    """True if a run_queue / train_cv process is running."""
    # Prefer PowerShell/CIM — wmic is removed on newer Windows.
    # Exclude powershell/cmd that merely *mention* the script names in a query.
    ps = (
        "$p = Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -and "
        "($_.Name -match 'python') -and "
        "($_.CommandLine -match 'run_queue\\.py|train_cv\\.py') -and "
        "($_.CommandLine -notmatch 'monitor_status\\.py') "
        "}; $p | Select-Object -ExpandProperty ProcessId"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            stderr=subprocess.DEVNULL, text=True, timeout=20,
        )
        return any(tok.strip().isdigit() for tok in out.split())
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where",
             "CommandLine like '%run_queue.py%' or CommandLine like '%train_cv.py%'",
             "get", "ProcessId,Name"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        return "python" in out.lower() and any(
            tok.strip().isdigit() for tok in out.split()
        )
    except Exception:
        return False


def handle_control_action(action: str) -> tuple[int, str]:
    """Apply pause / resume / start. Returns (http_status, message)."""
    from revision.control import write_control, read_control
    action = (action or "").strip().lower()
    if action == "pause":
        write_control("paused", "paused from dashboard")
        return 200, "PAUSED. Current epoch finishes, then training waits. Safe to shut down PC after status shows paused."
    if action in ("resume", "continue"):
        write_control("running", "resumed from dashboard")
        if not _trainer_alive():
            msg = start_training_queue("all10_best")
            return 200, f"Control set to RUNNING. Trainer was stopped — {msg}"
        return 200, "CONTINUED. Trainer will resume at the next pause check (after current epoch / before next fold)."
    if action == "start":
        write_control("running", "start from dashboard")
        msg = start_training_queue("all10_best")
        return 200, msg
    return 400, f"Unknown action: {action}. Use pause, resume, or start."


def start_training_queue(phase: str = "all10_best") -> str:
    """Launch run_queue detached if not already running."""
    if _trainer_alive():
        return "Training already running — not starting a second queue."
    from revision.control import write_control
    write_control("running", "started from dashboard")
    log = ROOT / "revision_results" / "queue_stdout.log"
    py = ROOT / "venv313" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    DETACHED_PROCESS = 0x00000008
    cmd = [str(py), "-u", str(ROOT / "revision" / "run_queue.py"), "--phase", phase]
    if phase == "all10_best":
        cmd += ["--best-strats", ",".join(BEST2)]
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n==== start {datetime.now()} phase={phase} ====\n")
        lf.flush()
        subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return f"Started training queue ({phase}). Log: {log}"


def fmt_cli(st: dict) -> str:
    c = st["current"]
    p = st["progress"]
    bar_len = 30
    filled = int(bar_len * p["percent"] / 100)
    bar = "#" * filled + "-" * (bar_len - filled)
    lines = [
        "=" * 72,
        f"  IEEE Revision Monitor  |  {st['updated_at']}  |  {st['phase']}",
        "=" * 72,
        f"  NOW:  {c['architecture'] or '?'}  |  strategy={c['strategy'] or '?'}"
        f"  |  fold={c['fold'] or '?'}  |  epoch={c['epoch'] or '?'}",
        f"  METRICS: {c['last_metrics'] or '(waiting for epoch log...)'}",
        "",
        f"  Progress folds: {p['folds_done']}/{p['folds_total']}  [{bar}] {p['percent']}%",
        f"  Jobs (model×strategy): {p['jobs_done']}/{p['jobs_total']}",
        f"  Avg fold time: {p['avg_min_per_fold']} min   ETA remaining: {p['eta']}",
        f"  Elapsed (since 1st ckpt): {p['elapsed_since_first_ckpt'] or 'n/a'}",
        "",
        "  Queue:",
    ]
    for j in st["queue"]:
        mark = {"done": "[DONE]", "running": "[>>>>]", "partial": "[part]", "pending": "[    ]"}[j["state"]]
        lines.append(
            f"    {mark} {j['display']:<16} {j['strategy']:<14} "
            f"{j['folds_done']}/{j['folds_total']} folds"
            + (f"  collapse={j['collapsed']}" if j["folds_done"] else "")
        )
    if st["recent_folds"]:
        lines += ["", "  Recent completed folds:"]
        for r in st["recent_folds"]:
            col = " COLLAPSED" if r["collapsed"] else ""
            lines.append(
                f"    {r['model']:<14} {r['strategy']:<12} {r['fold']}  "
                f"Acc={r['acc']:.3f} F1={r['f1']:.3f} Rec={r['recall']:.3f}{col}"
            )
    lines.append("=" * 72)
    return "\n".join(lines)


def html_page(st: dict) -> str:
    # Shell page — JS polls /api every 3s and re-renders (meta-refresh is unreliable).
    boot = json.dumps(st)
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
<title>Revision Training Monitor</title>
<style>
 body {{ font-family: Segoe UI, sans-serif; background:#0f1419; color:#e7ecf3; margin:24px; }}
 h1 {{ margin:0 0 8px; font-size:22px; }}
 .sub {{ color:#9aa7b8; margin-bottom:20px; }}
 .card {{ background:#1a2332; border-radius:12px; padding:16px 20px; margin-bottom:16px; }}
 .big {{ font-size:20px; font-weight:600; }}
 .bar {{ height:14px; background:#2a3548; border-radius:8px; overflow:hidden; margin:10px 0; }}
 .bar > div {{ height:100%; background:linear-gradient(90deg,#3d8bfd,#2dd4a8); width:0%; transition:width .4s; }}
 table {{ width:100%; border-collapse:collapse; font-size:14px; }}
 th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #2a3548; }}
 tr.done td {{ color:#7dcea0; }}
 tr.running td {{ color:#f0c674; font-weight:600; }}
 tr.partial td {{ color:#85c1e9; }}
 tr.pending td {{ color:#7f8c9b; }}
 .btn {{ border:0; border-radius:8px; padding:10px 16px; margin-right:8px; margin-top:8px;
         font-weight:600; cursor:pointer; color:#fff; }}
 .pause {{ background:#e67e22; }} .resume {{ background:#27ae60; }}
 .start {{ background:#3d8bfd; }} .badge {{ display:inline-block; padding:4px 10px;
         border-radius:999px; font-size:12px; margin-left:8px; }}
 .badge.on {{ background:#1e3a2f; color:#7dcea0; }}
 .badge.off {{ background:#3a1e1e; color:#f5b7b1; }}
 .badge.pause {{ background:#3a2e1e; color:#f0c674; }}
 .note {{ color:#9aa7b8; font-size:13px; line-height:1.45; margin-top:10px; }}
 .pulse {{ color:#2dd4a8; }}
</style>
</head><body>
<div id="app">Loading…</div>
<script>
const BOOT = {boot};
function esc(s) {{
  return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
async function send(action) {{
  const r = await fetch('/control?action=' + action, {{method:'POST', cache:'no-store'}});
  const t = await r.text();
  alert(t);
  await refresh();
}}
function render(st) {{
  const c = st.current || {{}};
  const p = st.progress || {{}};
  const ctrl = st.control || {{}};
  const alive = st.trainer_alive;
  const ctrlState = ctrl.state || 'running';
  const rows = (st.queue || []).map(j =>
    `<tr class="${{j.state}}"><td>${{esc(j.display)}}</td><td>${{esc(j.strategy)}}</td>` +
    `<td>${{j.folds_done}}/${{j.folds_total}}</td><td>${{esc(j.state)}}</td>` +
    `<td>${{j.collapsed || ''}}</td></tr>`
  ).join('');
  const recent = [...(st.recent_folds || [])].reverse().map(r =>
    `<tr><td>${{esc(r.model)}}</td><td>${{esc(r.strategy)}}</td><td>${{esc(r.fold)}}</td>` +
    `<td>${{Number(r.acc).toFixed(3)}}</td><td>${{Number(r.f1).toFixed(3)}}</td>` +
    `<td>${{Number(r.recall).toFixed(3)}}</td><td>${{r.collapsed ? 'YES' : ''}}</td></tr>`
  ).join('');
  document.getElementById('app').innerHTML = `
<h1>IEEE Revision Training Monitor</h1>
<div class="sub">${{esc(st.phase)}} · <span class="pulse">updated ${{esc(st.updated_at)}}</span> · live poll 3s
  <span class="badge ${{alive ? 'on' : 'off'}}">trainer ${{alive ? 'ALIVE' : 'STOPPED'}}</span>
  <span class="badge ${{ctrlState === 'paused' ? 'pause' : 'on'}}">control: ${{esc(ctrlState)}}</span>
</div>
<div class="card">
  <div class="big">Controls (safe — finished folds are never lost)</div>
  <button class="btn pause" onclick="send('pause')">Pause</button>
  <button class="btn resume" onclick="send('resume')">Continue</button>
  <button class="btn start" onclick="send('start')">Start / Resume queue after reboot</button>
  <div class="note">
    <b>Pause:</b> finishes the current epoch, then waits. Safe to shut down the PC after it shows PAUSED.<br/>
    <b>Continue:</b> unpauses a waiting trainer, or use with Start after reboot.<br/>
    <b>Start / Resume after reboot:</b> launches the queue again. Already-finished folds are skipped automatically.<br/>
    <b>Reliability:</b> completed folds + checkpoints on disk are kept. If power dies mid-fold, only that one unfinished fold restarts.
  </div>
</div>
<div class="card">
  <div class="big">Now training: ${{esc(c.architecture || '—')}} / ${{esc(c.strategy || '—')}}</div>
  <div>Fold ${{esc(c.fold ?? '—')}} · Epoch ${{esc(c.epoch ?? '—')}}${{c.status ? ' · ' + esc(c.status) : ''}}</div>
  <div style="margin-top:8px;color:#9aa7b8;font-size:13px">${{esc(c.last_metrics || '')}}</div>
</div>
<div class="card">
  <div>Folds ${{p.folds_done}} / ${{p.folds_total}} (${{p.percent}}%)
       · Jobs ${{p.jobs_done}}/${{p.jobs_total}}
       · Avg ${{p.avg_min_per_fold}} min/fold
       · ETA ${{esc(p.eta)}}
       · Elapsed ${{esc(p.elapsed_since_first_ckpt || 'n/a')}}</div>
  <div class="bar"><div style="width:${{p.percent || 0}}%"></div></div>
</div>
<div class="card">
  <h3 style="margin-top:0">Queue (Phase 3b)</h3>
  <table><tr><th>Model</th><th>Strategy</th><th>Folds</th><th>State</th><th>Collapsed</th></tr>
  ${{rows}}</table>
</div>
<div class="card">
  <h3 style="margin-top:0">Recent folds</h3>
  <table><tr><th>Model</th><th>Strategy</th><th>Fold</th><th>Acc</th><th>F1</th><th>Recall</th><th>Collapse</th></tr>
  ${{recent}}</table>
</div>`;
}}
async function refresh() {{
  try {{
    const r = await fetch('/api?t=' + Date.now(), {{cache:'no-store'}});
    const st = await r.json();
    render(st);
  }} catch (e) {{
    console.error(e);
  }}
}}
render(BOOT);
setInterval(refresh, 3000);
</script>
</body></html>"""


def run_cli(watch: bool, interval: int):
    while True:
        st = build_status()
        # also dump JSON for other tools
        out = ROOT / "revision_results" / "live_status.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(st, indent=2), encoding="utf-8")
        if watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(fmt_cli(st), flush=True)
        print(f"\n  Also writing: {out}", flush=True)
        if not watch:
            break
        time.sleep(interval)


def run_web(port: int):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            st = build_status()
            no_cache = [
                ("Cache-Control", "no-cache, no-store, must-revalidate"),
                ("Pragma", "no-cache"),
                ("Expires", "0"),
            ]
            if self.path.startswith("/api"):
                body = json.dumps(st, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in no_cache:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)
            else:
                body = html_page(st).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                for k, v in no_cache:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            if parsed.path != "/control":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"not found")
                return
            qs = parse_qs(parsed.query)
            action = (qs.get("action") or ["pause"])[0]
            code, msg = handle_control_action(action)
            body = msg.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    st = build_status()
    (ROOT / "revision_results" / "live_status.json").write_text(
        json.dumps(st, indent=2), encoding="utf-8"
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Monitor web UI: http://127.0.0.1:{port}", flush=True)
    print(f"JSON API:       http://127.0.0.1:{port}/api", flush=True)
    print("Pause / Continue / Start enabled. Ctrl+C to stop monitor only.", flush=True)
    server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description="Read-only training monitor")
    ap.add_argument("--watch", action="store_true", help="CLI live refresh")
    ap.add_argument("--interval", type=int, default=15, help="CLI refresh seconds")
    ap.add_argument("--web", action="store_true", help="Start local web dashboard")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.web:
        run_web(args.port)
    else:
        run_cli(args.watch, args.interval)


if __name__ == "__main__":
    main()
