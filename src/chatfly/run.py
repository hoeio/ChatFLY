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

from chatfly.chat import exp_chat
from chatfly.conditioning import exp_conditioning, token_patterns
from chatfly.fsm import exp_fsm_sweep, exp_multi_fly
from chatfly.grammar import exp_grammar
from chatfly.parity import exp_parity, exp_parity_table
from chatfly.sim import SIMULATORS
from chatfly.speak import exp_speak_baselines, exp_speak_fly
from chatfly.zumi import exp_zumi, exp_zumi_memory

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("chatfly")


def git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        return ""


def start_run(config_path: Path, seed: int, results_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = results_dir / f"{stamp}_{config_path.stem}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(config_path, run_dir / "config.yaml")  # verbatim; meta.json seed wins over it
    sha = git("rev-parse", "HEAD").strip() or "unknown"
    dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    if dirty:  # uncommitted changes: keep them so the run's code is recoverable
        (run_dir / "patch.diff").write_text(git("diff", "HEAD"))
    meta = {
        "seed": seed,
        "commit": sha + ("-dirty" if dirty else ""),
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


def exp_bench(cfg: dict, seed: int) -> dict:
    """Wall time of one present() of the MB model vs KC count (first call, with codegen, excluded)."""
    base = yaml.safe_load((ROOT / cfg["base"]).read_text())
    pats = None
    rows = []
    for kind in cfg["simulators"]:
        for n_kc in cfg["n_kc"]:
            model = {**base, "mb": {**base["mb"], "n_kc": n_kc}}
            t0 = time.perf_counter()
            sim = SIMULATORS[kind](model, seed)
            pats = token_patterns(model, np.random.default_rng(seed))
            sim.present(pats["A"], cfg["duration_ms"], [1.0, 0.0])
            build_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(cfg["repeats"]):
                sim.present(pats["A"], cfg["duration_ms"], [1.0, 0.0])
            step_s = (time.perf_counter() - t0) / cfg["repeats"]
            rows.append({"simulator": kind, "n_kc": n_kc, "build_and_first_s": build_s,
                         "present_s": step_s,
                         "realtime_factor": step_s / (cfg["duration_ms"] / 1000.0)})
            log.info("%s", rows[-1])
    return {"duration_ms": cfg["duration_ms"], "rows": rows}


EXPERIMENTS = {"bench": exp_bench, "conditioning": exp_conditioning, "parity": exp_parity,
               "parity_table": exp_parity_table, "fsm_sweep": exp_fsm_sweep,
               "multi_fly": exp_multi_fly, "chat": exp_chat, "zumi": exp_zumi,
               "zumi_memory": exp_zumi_memory, "speak_baselines": exp_speak_baselines,
               "speak_fly": exp_speak_fly, "grammar": exp_grammar}


def main(argv=None) -> Path:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    a = p.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())
    seed = cfg["seed"] if a.seed is None else a.seed
    run_dir = start_run(a.config, seed, a.results)
    status = "failed"
    try:
        metrics = EXPERIMENTS[cfg["experiment"]](cfg, seed)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        status = "done"
    finally:
        (run_dir / "status").write_text(status + "\n")
        log.info("%s -> %s", status, run_dir)
        for h in logging.getLogger().handlers:
            h.close()
    return run_dir


if __name__ == "__main__":
    main()
