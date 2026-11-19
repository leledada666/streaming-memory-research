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

# v2 smoke: position fix + lag=0 positive control + position-free ablation
CONFIGS = [
    ("crossmem_pos", 0, 32, 4),
    ("crossmem_pos", 10, 32, 4),
    ("crossmem_pos", 60, 96, 2),
    ("crossmem_nopos", 0, 32, 4),
    ("crossmem_nopos", 10, 32, 4),
    ("crossmem_nopos", 60, 96, 2),
    ("bypass", 0, 32, 4),
    ("bypass", 10, 32, 4),
    ("bypass", 60, 96, 2),
    ("bypass_wide", 10, 32, 4),
    ("crossmem_slots_pos", 10, 32, 4),
    ("crossmem_slots_nopos", 10, 32, 4),
]

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
summary = []
for arm, lag, T, B in CONFIGS:
    log = LOG / f"v2_{arm}_L{lag}_B{B}_T{T}.txt"
    cmd = [PY, str(SCRIPT), "--arm", arm, "--lag", str(lag), "--T", str(T), "--B", str(B),
           "--steps", "200", "--eval_every", "40", "--seed", "0", "--threads", "6",
           "--val_batches", "2", "--test_batches", "3", "--probe_batches", "2",
           "--val_B", str(48 if T >= 96 else 64), "--tag", "v2smoke"]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    txt = log.read_text(encoding="utf-8", errors="replace")
    params = None
    done = None
    for line in txt.splitlines():
        if line.startswith("params: "):
            params = json.loads(line[len("params: "):])
        if line.startswith("DONE "):
            done = json.loads(line[len("DONE "):])
    summary.append({"arm": arm, "lag": lag, "T": T, "B": B, "rc": rc,
                    "trainable_params": (params or {}).get("trainable_total"),
                    "result": done, "runtime_s": round(time.time() - t0, 1), "log": str(log)})
    print(json.dumps(summary[-1], ensure_ascii=False), flush=True)

(LOG / "v2_smoke_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("ALL_DONE", flush=True)
for s in summary:
    r = s["result"] or {}
    print(s["arm"], "L", s["lag"], "p", s["trainable_params"],
          "val", r.get("val_acc_best"), "test", r.get("test_acc"),
          "rc", s["rc"], "hash", r.get("backbone_hash_unchanged"), flush=True)