import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
SCRIPT = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\m3_crossmem.py")
LOG = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\m3_crossmem")
LOG.mkdir(parents=True, exist_ok=True)
OUT = Path(r"C:\Users\lele\AppData\Local\Temp\opencode")

# CPU smoke grid: lag 10 at T=32, lag 60 needs T>lag -> T=96 with B=2
CONFIGS = [
    ("bypass", 10, 32, 4, 200),
    ("bypass", 60, 96, 2, 200),
    ("crossmem", 10, 32, 4, 200),
    ("crossmem", 60, 96, 2, 200),
    ("crossmem_slots", 10, 32, 4, 200),
    ("crossmem_slots", 60, 96, 2, 200),
]

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
summary = []
for arm, lag, T, B, steps in CONFIGS:
    log = LOG / f"log_{arm}_L{lag}_B{B}_T{T}.txt"
    cmd = [PY, str(SCRIPT), "--arm", arm, "--lag", str(lag), "--T", str(T), "--B", str(B),
           "--steps", str(steps), "--eval_every", "40", "--seed", "0", "--threads", "6",
           "--val_B", str(48 if T >= 96 else 64), "--tag", "smoke"]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    txt = log.read_text(encoding="utf-8", errors="replace")
    params = None
    best = None
    for line in txt.splitlines():
        if line.startswith("params: "):
            params = json.loads(line[len("params: "):])
        if line.startswith("DONE "):
            best = json.loads(line[len("DONE "):])
    summary.append({"arm": arm, "lag": lag, "T": T, "B": B, "rc": rc,
                    "trainable_params": (params or {}).get("trainable_total"),
                    "result": best, "runtime_s": round(time.time() - t0, 1), "log": str(log)})
    print(json.dumps(summary[-1]), flush=True)

(LOG / "smoke_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("ALL_DONE", flush=True)
for s in summary:
    r = s["result"] or {}
    print(s["arm"], "L", s["lag"], "params", s["trainable_params"],
          "val_best", r.get("val_acc_best"), "ok", s["rc"] == 0,
          "hash_ok", r.get("backbone_hash_unchanged"), flush=True)