import json
from pathlib import Path

import numpy as np
import yaml

from chatfly.conditioning import run_seed
from chatfly.run import main
from chatfly.sim import RateSimulator, SpikingSimulator

CFG = Path(__file__).resolve().parents[1] / "configs" / "phase1_conditioning.yaml"


def small_cfg(**over):
    cfg = yaml.safe_load(CFG.read_text())
    cfg["mb"]["n_kc"] = 400
    cfg["spiking"]["codegen_target"] = "numpy"
    cfg.update(over)
    return cfg


def test_learn_rule():
    sim = RateSimulator(small_cfg(), 0)
    kc = np.zeros(sim.n_kc)
    kc[:10] = 100.0
    sim.learn(kc, np.array([1.0, 0.0]))  # DAN in compartment 0 only
    w0, ed, er = sim.w0, sim.eta_dep, sim.eta_rec
    assert np.allclose(sim.w[0, :10], w0 * (1 - ed))  # coincidence -> depression
    assert np.allclose(sim.w[0, 10:], w0) and np.allclose(sim.w[1], w0)  # inactive / at w0
    sim.learn(kc, np.array([0.0, 0.0]))  # KC active, DAN silent -> recover toward w0
    assert np.allclose(sim.w[0, :10], w0 - (1 - er) * w0 * ed)


def test_rate_conditioning_signs():
    out = run_seed("rate", small_cfg(), 0)
    assert out["post"]["A"] > 0 > out["post"]["B"]  # A+ -> approach, B- -> avoid
    assert abs(out["ext"]["A"]) < out["post"]["A"]  # extinction weakens A
    assert out["rev"]["A"] < 0 < out["rev"]["B"]  # reversal flips


def test_spiking_kc_sparse_and_state_reset():
    sim = SpikingSimulator(small_cfg(), 0)
    r = np.full(sim.n_pn, 2.0)
    r[:38] = 200.0
    a = sim.present(r, 200.0)["kc"]
    b = sim.present(r, 200.0)["kc"]
    for x in (a, b):
        assert 0.0 < (x >= sim.kc_active_hz).mean() < 0.3
    assert sim.present(np.full(sim.n_pn, 2.0), 200.0)["kc"].sum() == 0  # no carry-over


def test_run_writes_artifacts(tmp_path):
    cfg = small_cfg(n_seeds=2, simulators=["rate"])
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_dir = main([str(path), "--seed", "3", "--results", str(tmp_path / "res")])
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["seed"] == 3 and meta["commit"]
    assert (run_dir / "status").read_text().strip() == "done"
    assert "rate" in json.loads((run_dir / "metrics.json").read_text())
    if meta["commit"].endswith("-dirty"):
        assert (run_dir / "patch.diff").exists()


def test_parity_rate_seed():
    from chatfly.parity import delta, run_seed

    assert [delta(s, x) for s in ("even", "odd") for x in "AB"] == ["odd", "even", "even", "odd"]
    cfg = yaml.safe_load((CFG.parent / "phase2_parity.yaml").read_text())
    cfg["mb"]["n_kc"] = 400
    cfg["loop"].update(lengths=[10], n_seq=2)
    out = run_seed(("rate", cfg, 0))
    assert out["step_acc"] == 1.0 and out["loop"]["10"]["seq_ok"] == 1.0
    assert out["controls"]["teacher_shuffle"] < 1.0


def test_flywire_wiring_and_controls():
    import pytest

    from chatfly.sim import build_wiring

    cfg = yaml.safe_load((CFG.parent / "phase3_parity_flywire.yaml").read_text())
    data = Path(__file__).resolve().parents[1] / cfg["connectome"]["dir"]
    if not (data / cfg["connectome"]["connections"]).exists():
        pytest.skip("FlyWire data not downloaded (data/README.md)")
    real = build_wiring(cfg, np.random.default_rng(0))
    assert real["n_kc"] > 2000 and real["n_pn"] > 100 and real["mask"] is not None
    np.testing.assert_allclose(real["mask"].sum(1), real["n_kc"])  # per-MBON drive normalized
    cfg["mb"]["wiring"] = "shuffle"
    sh = build_wiring(cfg, np.random.default_rng(0))
    for a, b in [("kc_edge", "kc_edge"), ("pn_edge", "pn_edge")]:  # degrees preserved
        assert (np.bincount(real[a]) == np.bincount(sh[b], minlength=len(np.bincount(real[a])))).all()
    assert len(set(zip(sh["kc_edge"], sh["pn_edge"]))) == len(sh["kc_edge"])  # no duplicates
    assert (real["pn_edge"] != sh["pn_edge"]).mean() > 0.5
