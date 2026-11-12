import json
from pathlib import Path

import numpy as np
import yaml

from chatfly.run import main
from chatfly.sim import RateSimulator, SpikingSimulator

CFG = Path(__file__).resolve().parents[1] / "configs" / "phase0_empty.yaml"


def test_rate_and_spiking_agree_without_noise():
    model = {**yaml.safe_load(CFG.read_text())["model"], "n_neurons": 5, "codegen_target": "numpy"}
    stim = np.array([0.0, 15.0, 25.0, 30.0, 40.0])  # threshold gap is 20 mV
    rate = RateSimulator(model, 0).present(stim, 1000.0)["main"]
    spk = SpikingSimulator(model, 0).present(stim, 1000.0)["main"]
    assert rate[0] == rate[1] == spk[0] == spk[1] == 0
    np.testing.assert_allclose(spk[2:], rate[2:], rtol=0.05)


def test_run_writes_artifacts(tmp_path):
    run_dir = main([str(CFG), "--seed", "3", "--results", str(tmp_path)])
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["seed"] == 3 and meta["commit"]
    assert (run_dir / "config.yaml").read_text() == CFG.read_text()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {"rate", "spiking"}
    assert (run_dir / "run.log").stat().st_size > 0
