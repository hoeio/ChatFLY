"""FlyWire v783 mushroom-body extraction (Phase 3). Data: data/README.md.

`load_mb(conn_cfg)` returns, for one hemisphere:
  n_pn, n_kc, pn_edge/kc_edge (uniglomerular PN -> KC edges with >= min_synapses),
  kc_mbon (n_mbon, n_kc) synapse counts, mbon_type (n_mbon,), kc_to_apl / apl_to_kc (n_kc,)
  synapse counts, kc_kc (n_kc, n_kc) synapse counts (not used by the model yet), and ids.
Connection counts are summed over neuropils. The result is cached as .npz next to the data.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_mb(conn_cfg: dict) -> dict:
    return _load(json.dumps(conn_cfg, sort_keys=True))


@lru_cache(maxsize=4)
def _load(key: str) -> dict:
    c = json.loads(key)
    d = ROOT / c["dir"]
    cache = d / f"mb_{c['side']}_min{c['min_synapses']}.npz"
    if not cache.exists():
        np.savez_compressed(cache, **_extract(c, d))
    z = np.load(cache, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["n_pn"], out["n_kc"] = len(out["pn_ids"]), len(out["kc_ids"])
    out["mbon_type"] = [str(t) for t in out["mbon_type"]]
    return out


def _extract(c: dict, d: Path) -> dict:
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    import pyarrow.feather as pf

    ann = pcsv.read_csv(d / c["annotations"],
                        parse_options=pcsv.ParseOptions(delimiter="\t"),
                        convert_options=pcsv.ConvertOptions(column_types={"root_id": "int64"}))
    a = {k: np.array(ann[k].to_pylist(), dtype=object)
         for k in ("cell_class", "cell_sub_class", "cell_type", "side")}
    rid = ann["root_id"].to_numpy()
    side = a["side"] == c["side"]
    pn_ids = np.sort(rid[side & (a["cell_class"] == "ALPN")
                         & (a["cell_sub_class"] == "uniglomerular")])
    kc_all = np.sort(rid[side & (a["cell_class"] == "Kenyon_Cell")])
    mbon_mask = side & (a["cell_class"] == "MBON")
    order = np.argsort(rid[mbon_mask])
    mbon_ids, mbon_type = rid[mbon_mask][order], a["cell_type"][mbon_mask][order]
    apl_ids = rid[side & (a["cell_type"] == "APL")]
    assert len(apl_ids) == 1, apl_ids

    t = pf.read_table(d / c["connections"], columns=[c["col_pre"], c["col_post"], c["col_syn"]])
    pre_set = np.concatenate([pn_ids, kc_all, apl_ids])
    post_set = np.concatenate([kc_all, mbon_ids, apl_ids])
    import pyarrow as pa

    keep = pc.and_(pc.is_in(t[c["col_pre"]], value_set=pa.array(pre_set)),
                   pc.is_in(t[c["col_post"]], value_set=pa.array(post_set)))
    t = t.filter(keep)
    pre, post = t[c["col_pre"]].to_numpy(), t[c["col_post"]].to_numpy()
    syn = t[c["col_syn"]].to_numpy().astype(np.int64)

    def matrix(rows, cols):  # counts summed over neuropils; rows = post, cols = pre
        ri = {r: i for i, r in enumerate(rows)}
        ci = {r: i for i, r in enumerate(cols)}
        m = np.zeros((len(rows), len(cols)), np.int64)
        for p, q, s in zip(pre, post, syn, strict=True):
            if q in ri and p in ci:
                m[ri[q], ci[p]] += s
        return m

    pn_kc = matrix(kc_all, pn_ids)  # (n_kc_all, n_pn)
    claws = (pn_kc >= c["min_synapses"]).sum(1)
    olf = claws >= c["kc_min_claws"]  # KCs with uniglomerular-PN claws (drops e.g. visual KCs)
    kc_ids = kc_all[olf]
    kc_edge, pn_edge = np.nonzero(pn_kc[olf] >= c["min_synapses"])
    return {
        "pn_ids": pn_ids, "kc_ids": kc_ids, "mbon_ids": mbon_ids,
        "mbon_type": np.array(mbon_type, dtype=str), "apl_id": apl_ids,
        "pn_edge": pn_edge, "kc_edge": kc_edge,
        "pn_kc_syn": pn_kc[olf],
        "kc_mbon": matrix(mbon_ids, kc_ids),
        "kc_to_apl": matrix(apl_ids, kc_ids)[0],
        "apl_to_kc": matrix(kc_ids, apl_ids)[:, 0],
        "kc_kc": matrix(kc_ids, kc_ids),
        "n_kc_dropped": np.array(int((~olf).sum())),
    }
