#!/usr/bin/env python3
"""
progress.py — lightweight watcher for the nn_expert.py GRU training run.

Stdlib only (no torch/numpy import, so it never competes for training CPU).

Usage:
  python progress.py                     # one-shot status
  watch -n 60 python progress.py         # live status every minute

Epoch timing is calibrated from the first nn_gru_best.pt save (mtime of the
earliest observed checkpoint). Run this at least once during epoch 1 for an
accurate ETA; before that it can only show process health.
"""

import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HEDGE_RESULTS",
                     os.path.join(HERE, "benchmarks", "results"))
CKPT = os.path.join(OUT, "nn_gru_best.pt")
STATE = os.path.join(OUT, ".progress_state.json")
EPOCHS = 6
SCORING_MIN = 8          # rough allowance: aligned scoring of valid+test


def find_trainee():
    out = subprocess.run(["ps", "-eo", "pid,etime,time,%cpu,cmd"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "nn_expert.py" in line:
            f = line.split()
            return {"pid": f[0], "etime": f[1], "cputime": f[2],
                    "cpu": float(f[3])}
    return None


def etime_seconds(s):
    def sec(x):
        if "-" in x:
            d, rest = x.split("-")
            return int(d) * 86400 + sec(rest)
        bits = [int(b) for b in x.split(":")]
        while len(bits) < 3:
            bits.insert(0, 0)
        return bits[0] * 3600 + bits[1] * 60 + bits[2]
    return sec(s)


def main():
    now = time.time()
    p = find_trainee()
    state = {}
    if os.path.exists(STATE):
        try:
            with open(STATE) as f:
                state = json.load(f)
        except (ValueError, OSError):
            state = {}

    print("== GRU training progress ==")
    if p is None:
        print("training process: NOT RUNNING")
        elapsed = None
    else:
        elapsed = etime_seconds(p["etime"])
        print(f"process:    pid {p['pid']}  wall {p['etime']}  "
              f"cpu {p['cputime']}  load {p['cpu']:.0f}%")
        print("live epoch lines: check the training terminal (pts/0)")

    obs = state.get("mtimes", [])
    if os.path.exists(CKPT):
        mt = os.path.getmtime(CKPT)
        if mt not in obs:
            obs.append(mt)
            obs.sort()
        state["mtimes"] = obs
        print(f"checkpoint: nn_gru_best.pt present, "
              f"{(now - mt) / 60:.0f} min old, "
              f"{os.path.getsize(CKPT) / 1e6:.0f} MB")
    else:
        print("checkpoint: not saved yet (epoch 1 in progress)")

    epoch1 = state.get("epoch1_wall")
    if epoch1 is None and elapsed is not None and obs and p is not None:
        start_ts = now - elapsed
        state["epoch1_wall"] = obs[0] - start_ts
        epoch1 = state["epoch1_wall"]

    for name in ("nn_logp_valid.npy", "nn_logp_test.npy"):
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            ts = time.strftime("%H:%M", time.localtime(os.path.getmtime(path)))
            print(f"{name}: saved {ts}")
        else:
            print(f"{name}: pending")

    if epoch1 and elapsed is not None:
        total = epoch1 * EPOCHS
        pos = elapsed / epoch1
        print(f"epoch pace: ~{epoch1 / 60:.0f} min/epoch  "
              f"-> ~{pos:.1f}/{EPOCHS} epochs done (est)")
        if pos >= EPOCHS:
            print("training loop done; scoring phase or finished "
                  "(see nn_logp_*.npy above)")
        else:
            remain = total - elapsed
            fin = time.strftime("%H:%M", time.localtime(
                now + remain + SCORING_MIN * 60))
            print(f"ETA: ~{remain / 60:.0f} min training "
                  f"+ ~{SCORING_MIN} min scoring -> done ~{fin}")
    elif elapsed is not None:
        print("ETA: unknown until epoch-1 checkpoint is observed "
              "(run this script again right after it appears)")

    try:
        with open(STATE, "w") as f:
            json.dump(state, f)
    except OSError:
        pass


if __name__ == "__main__":
    main()
