import json, subprocess, sys, time
from pathlib import Path
PY = sys.executable
S = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\m3_crossmem.py")
D = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\m3_crossmem")
jobs = [
    ("crossmem_nopos", 10, 32, 8, 400, 100),
    ("crossmem_pos", 60, 96, 4, 600, 100),
]
for arm, lag, T, B, steps, ev in jobs:
    log = D / f"long2_{arm}_L{lag}.txt"
    cmd = [PY, str(S), "--arm", arm, "--lag", str(lag), "--T", str(T), "--B", str(B),
           "--steps", str(steps), "--eval_every", str(ev), "--val_batches", "2",
           "--test_batches", "3", "--probe_batches", "2", "--threads", "6",
           "--tag", "longsmoke2"]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n"); f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    txt = log.read_text(encoding="utf-8", errors="replace")
    done = [l for l in txt.splitlines() if l.startswith("DONE ")]
    print(arm, lag, "rc", rc, "s", round(time.time()-t0), done[-1] if done else "NO_DONE", flush=True)
print("LONG2_DONE")