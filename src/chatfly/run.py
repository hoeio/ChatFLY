"""Experiment runner: `python -m chatfly.run configs/<name>.yaml [--seed N]`.

Every run writes results/<run_id>/ with config.yaml, meta.json (seed, commit,
versions), metrics.json and run.log.
"""

import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from chatfly.sim import SIMULATORS

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("chatfly")


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty.strip() else "")


def start_run(config_path: Path, seed: int, results_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = results_dir / f"{stamp}_{config_path.stem}_s{seed}"
    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.yaml")
    meta = {
        "seed": seed,
        "commit": git_commit(),
        "config": str(config_path),
        "started": stamp,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import brian2

        meta["brian2"] = brian2.__version__
    except ImportError:
        meta["brian2"] = None
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(run_dir / "run.log"), logging.StreamHandler()],
        force=True,
    )
    log.info("run %s seed=%d commit=%s", run_dir.name, seed, meta["commit"])
    return run_dir


def exp_empty(cfg: dict, seed: int) -> dict:
    """Drive an unconnected population with random constant input; report rates per simulator."""
    rng = np.random.default_rng(seed)
    model = cfg["model"]
    lo, hi = cfg["stimulus"]["drive_range_mv"]
    stim = rng.uniform(lo, hi, model["n_neurons"])
    out = {}
    for kind in cfg["simulators"]:
        t0 = time.perf_counter()
        rates = SIMULATORS[kind](model, seed).present(stim, cfg["stimulus"]["duration_ms"])["main"]
        out[kind] = {
            "mean_rate_hz": float(rates.mean()),
            "frac_active": float((rates > 0).mean()),
            "wall_s": time.perf_counter() - t0,
        }
        log.info("%s: %s", kind, out[kind])
    return out


def exp_bench(cfg: dict, seed: int) -> dict:
    """Wall time of one present() call vs population size (build excluded, warm-up excluded)."""
    rng = np.random.default_rng(seed)
    lo, hi = cfg["stimulus"]["drive_range_mv"]
    dur = cfg["stimulus"]["duration_ms"]
    rows = []
    for kind in cfg["simulators"]:
        for n in cfg["sizes"]:
            model = {**cfg["model"], "n_neurons": n}
            t0 = time.perf_counter()
            sim = SIMULATORS[kind](model, seed)
            stim = rng.uniform(lo, hi, n)
            sim.present(stim, dur)  # warm-up: includes code generation / compilation
            build_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(cfg["repeats"]):
                sim.present(stim, dur)
            step_s = (time.perf_counter() - t0) / cfg["repeats"]
            rows.append({"simulator": kind, "n": n, "build_and_first_s": build_s,
                         "present_s": step_s, "realtime_factor": step_s / (dur / 1000.0)})
            log.info("%s", rows[-1])
    return {"duration_ms": dur, "rows": rows}


EXPERIMENTS = {"empty": exp_empty, "bench": exp_bench}


def main(argv=None) -> Path:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    a = p.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())
    seed = cfg["seed"] if a.seed is None else a.seed
    run_dir = start_run(a.config, seed, a.results)
    metrics = EXPERIMENTS[cfg["experiment"]](cfg, seed)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("done -> %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
