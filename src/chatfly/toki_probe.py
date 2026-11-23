"""Phase 9 feasibility probes (ADR 0014), selection seeds only.

probe9_memory: can a memory hold facts about any (subject phrase, predicate phrase) without
predicate routing, so that known facts are answered and unknown pairs are declined (the 8d-2 /
8e-2 / 8i failure)? Compares 8i's union smell with chained binding (toki.Binder), plus controls.
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from chatfly.toki import CONTENT, Binder, Fly, Odors

log = logging.getLogger("chatfly")


def world(rng, m):
    """Subjects with shared heads and modifiers (a conjunction is needed), predicates of 1-2 words,
    facts (S, P, polarity) with changes of mind, and the questions with their answers."""
    words = list(rng.permutation([w for w in CONTENT if w not in ("mi", "sina")]))
    heads, mods, pw = words[:m["n_heads"]], words[m["n_heads"]:m["n_heads"] + m["n_mods"]], words[40:]
    subs = [(h,) for h in heads] + [(h, x) for h in heads for x in mods]
    subs += [(h, x, y) for h in heads for x in mods for y in mods if x != y]
    subs = [subs[i] for i in rng.permutation(len(subs))]
    known, unknown = subs[:m["n_subj"]], subs[m["n_subj"]:m["n_subj"] + m["n_unknown"]]
    preds = [(w,) for w in pw[:m["n_pred"] // 2]] + [(pw[i], pw[i + 1]) for i in range(m["n_pred"] // 2, m["n_pred"], 1)]
    pairs = [(s, p) for s in known for p in preds]
    pairs = [pairs[i] for i in rng.permutation(len(pairs))[:m["n_facts"]]]
    told, truth = [], {}
    for s, p in pairs:
        v = bool(rng.integers(2))
        told.append((s, p, v))
        truth[(s, p)] = v
    for i in rng.permutation(len(pairs))[:m["n_change"]]:  # changes of mind, told later
        s, p = pairs[i]
        truth[(s, p)] = not truth[(s, p)]
        told.append((s, p, truth[(s, p)]))
    asked = set(pairs)
    q_unknown = [(s, preds[int(rng.integers(len(preds)))]) for s in unknown]
    q_untold = [(s, p) for s in known for p in preds if (s, p) not in asked]
    q_untold = [q_untold[i] for i in rng.permutation(len(q_untold))[:len(q_unknown)]]
    return told, truth, q_unknown + q_untold


def _job(args):
    method, cfg, seed = args
    m, t0 = cfg["probe"], time.time()
    rng = np.random.default_rng([seed, 9, 1])
    told, truth, q_none = world(rng, m)
    od = Odors(cfg, seed, cfg["mb"]["n_pn"], cfg["tokens"]["pn_active_frac"])
    if method.startswith("union"):
        f = {**m[method if method in m else "union"], "n_pn": od.n_pn}
        def key(s, p):
            return np.max([od("w", w) for w in s + p], axis=0)
    else:
        f = {**m["key_memory"], "n_pn": od.n_pn}
        b = Binder(cfg, seed, od, m["binder"], "no_kc" if method == "chain_binder_no_kc" else None)
        def key(s, p):
            return b(b.phrase("subj", s), b.phrase("pred", p))
    ctl = "no_kc" if method in ("union_no_kc", "chain_memory_no_kc") else None
    mem = Fly(cfg, seed, 1, [True, False], f, ctl)
    for s, p, v in told:
        mem.teach(key(s, p), v, m["n_teach"])
    known = [(mem.read(key(s, p)), v) for (s, p), v in truth.items()]
    none = [mem.read(key(s, p)) for s, p in q_none]
    out = {"method": method, "seed": seed, "s": round(time.time() - t0, 1)}
    for thr in m["thresholds"]:
        out[f"known@{thr}"] = float(np.mean([a == v and c >= thr for (a, c), v in known]))
        out[f"declined@{thr}"] = float(np.mean([a is None or c < thr for a, c in none]))
        out[f"both@{thr}"] = min(out[f"known@{thr}"], out[f"declined@{thr}"])
    log.info("%s", out)
    return out


def with_base(cfg):
    return {**yaml.safe_load((Path(__file__).resolve().parents[2] / cfg["base"]).read_text()), **cfg}


def exp_probe9_memory(cfg: dict, seed: int) -> dict:
    cfg = with_base(cfg)
    m = cfg["probe"]
    jobs = [(meth, cfg, s) for meth in m["methods"] for s in cfg["select_seeds"]]
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        rows = list(ex.map(_job, jobs))
    summary = {}
    for meth in m["methods"]:
        rs = [r for r in rows if r["method"] == meth]
        summary[meth] = {k: [round(float(np.mean([r[k] for r in rs])), 3), round(float(np.min([r[k] for r in rs])), 3)]
                         for k in rs[0] if "@" in k}
        best = max(m["thresholds"], key=lambda t: summary[meth][f"both@{t}"][1])
        log.info("%s best threshold %s: %s", meth, best, {k: v for k, v in summary[meth].items() if k.endswith(f"@{best}")})
    return {"rows": rows, "summary": summary}


def tatoeba_clauses():
    from chatfly.toki import tokens

    path = Path(__file__).resolve().parents[2] / "data/tatoeba/tok_sentences.tsv"
    return [c for line in path.read_text(encoding="utf8").splitlines() for c in tokens(line.split("\t")[2]) if c]


def _rjob(args):
    from chatfly.toki import NAME, Recall, slots

    variant, n_facts, cfg, seed = args
    m, t0 = cfg["recall"], time.time()
    if not CLAUSES:  # worker processes start empty (spawn)
        CLAUSES.extend(tatoeba_clauses())
    rng = np.random.default_rng([seed, 9, 3])
    facts, seen = [], set()
    for i in rng.permutation(len(CLAUSES)):  # real sentences; one fact per (S, P) and per (S, li)
        f = slots(CLAUSES[i])
        if f and (f[0], f[1]) not in seen and (f[0], "li") not in seen:
            seen |= {(f[0], f[1]), (f[0], "li")}
            facts.append(f)
        if len(facts) == n_facts + m["n_unknown"]:
            break
    told, unknown = facts[:n_facts], facts[n_facts:]
    who = {}  # (P, O) -> every told subject: "seme li P e O?" may have several right answers
    for s, p, o, _ in told:
        who.setdefault((p, o), set()).add(s)
    od = Odors(cfg, seed, m.get("n_pn", cfg["mb"]["n_pn"]), m.get("pn_active_frac", cfg["tokens"]["pn_active_frac"]))
    bc = {**m["binder"], **m.get("noisy_binder", {}), "poisson": True} if variant.startswith("noisy") else m["binder"]
    b = Binder(cfg, seed, od, bc, "no_kc" if variant == "binder_no_kc" else None)
    f = {**m["memory"], "n_pn": od.n_pn, **({"code": tuple(m["code"])} if variant != "onehot" else {})}
    mem = Recall(cfg, seed, b, f, CONTENT + [NAME], "no_kc" if variant == "memory_no_kc" else None)

    def k(*parts):  # one key per question kind: (S, P) -> O ; (S, li) -> P ; (who, P, O) -> S
        return [b.phrase(role, ws) for role, ws in parts]

    def key(kind, s, p, o):
        ks = {"obj": k(("subj", s), ("pred", p), ("q", ("e",))), "pred": k(("subj", s), ("q", ("li",))),
              "subj": k(("q", ("seme",)), ("pred", p), ("obj", o or ("-",)))}[kind]
        h = ks[0]
        for x in ks[1:]:
            h = b(h, x)
        return h
    for s, p, o, _ in told:
        for kind, ans in (("obj", o), ("pred", p), ("subj", s)):
            mem.tell(key(kind, s, p, o), ans, m["n_teach"])
    out = {"variant": variant, "n_facts": n_facts, "seed": seed}
    for thr in m["thresholds"]:
        for kind, j in (("obj", 2), ("pred", 1)):
            out[f"{kind}@{thr}"] = float(np.mean([mem.ask(key(kind, *x[:3]), thr) == x[j] for x in told]))
        out[f"subj@{thr}"] = float(np.mean([mem.ask(key("subj", *x[:3]), thr) in who[(x[1], x[2])] for x in told]))
        qs = [(kind, x) for x in unknown for kind in ("obj", "pred", "subj") if kind != "subj" or (x[1], x[2]) not in who]
        out[f"declined@{thr}"] = float(np.mean([mem.ask(key(kind, *x[:3]), thr) is None for kind, x in qs]))
        out[f"all@{thr}"] = min(out[f"{q}@{thr}"] for q in ("obj", "pred", "subj", "declined"))
    out["s"] = round(time.time() - t0, 1)
    log.info("%s", {a: (round(v, 3) if isinstance(v, float) else v) for a, v in out.items() if "@" not in a or a.endswith(f"@{m['thresholds'][2]}")})
    return out


CLAUSES = []


def exp_probe9_recall(cfg: dict, seed: int) -> dict:
    cfg = with_base(cfg)
    m = cfg["recall"]
    jobs = [(v, n, cfg, s) for v in m["variants"] for n in m["n_facts"] for s in cfg["select_seeds"]]
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        rows = list(ex.map(_rjob, jobs))
    summary = {}
    for v in m["variants"]:
        for n in m["n_facts"]:
            rs = [r for r in rows if r["variant"] == v and r["n_facts"] == n]
            best = max(m["thresholds"], key=lambda t: min(r[f"all@{t}"] for r in rs))
            summary[f"{v}/{n}"] = {"threshold": best, **{q: [round(float(np.mean([r[f"{q}@{best}"] for r in rs])), 3),
                                                             round(float(np.min([r[f"{q}@{best}"] for r in rs])), 3)]
                                                         for q in ("obj", "pred", "subj", "declined", "all")}}
            log.info("%s/%s %s", v, n, summary[f"{v}/{n}"])
    return {"rows": rows, "summary": summary}


def _gjob(args):
    from chatfly.toki import ACCEPT, CLASSES, GRAMMAR, klass, parse

    setting, cfg, seed = args
    g, t0 = cfg["grammar"], time.time()
    if not CLAUSES:
        CLAUSES.extend(tatoeba_clauses())
    f = {**g["settings"][setting.removesuffix("_no_kc")], "n_pn": cfg["mb"]["n_pn"]}
    od = Odors(cfg, seed + 500, cfg["mb"]["n_pn"], cfg["tokens"]["pn_active_frac"])
    states = list(GRAMMAR)
    fly = Fly(cfg, seed, 3, states + ["X"], f, "no_kc" if setting.endswith("_no_kc") else None)
    table = [(s, c, GRAMMAR[s].get(c, "X")) for s in states for c in CLASSES + ("EOS",) if c != "EOS"]
    table += [(s, "EOS", "S0" if s in ACCEPT else "X") for s in states]  # end of clause: accepted -> S0
    def x(s, c):
        return np.maximum(od("s", s), od("c", c))
    rng = np.random.default_rng([seed, 9, 4])
    for _ in range(g["epochs"]):
        for i in rng.permutation(len(table)):
            fly.teach(x(*table[i][:2]), table[i][2])
    for _ in range(g["rehearse"]):
        for i in rng.permutation(len(table)):
            s, c, t = table[i]
            if fly.read(x(s, c))[0] != t:
                fly.teach(x(s, c), t)
    step = float(np.mean([fly.read(x(s, c))[0] == t for s, c, t in table]))
    def run(ws):
        s, out = "S0", []
        for w in ws + ["EOS"]:
            s = fly.read(x(s, klass(w) if w != "EOS" else "EOS"))[0] or "X"
            if s == "X":
                return None
            out.append(s)
        return out[:-1]
    sample = [CLAUSES[i] for i in rng.permutation(len(CLAUSES))[:g["n_test"]]]
    agree = float(np.mean([run(c) == parse(c) for c in sample]))
    rejected = [c for c in sample if parse(c) is None]
    salad = [list(rng.choice(sample[i % len(sample)], len(sample[i % len(sample)]), replace=False)) for i in range(g["n_test"])]
    salad = [c for c in salad if parse(c) is None]
    out = {"setting": setting, "seed": seed, "table": step, "tatoeba_agree": agree,
           "salad_rejected": float(np.mean([run(c) is None for c in salad])),
           "n_ref_rejected": len(rejected), "s": round(time.time() - t0, 1)}
    log.info("%s", out)
    return out


def exp_probe9_grammar(cfg: dict, seed: int) -> dict:
    cfg = with_base(cfg)
    g = cfg["grammar"]
    jobs = [(st, cfg, s) for st in g["run"] for s in cfg["select_seeds"]]
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        rows = list(ex.map(_gjob, jobs))
    summary = {st: {k: [round(float(np.mean([r[k] for r in rows if r["setting"] == st])), 4),
                        round(float(np.min([r[k] for r in rows if r["setting"] == st])), 4)]
                    for k in ("table", "tatoeba_agree", "salad_rejected")} for st in g["run"]}
    for st, v in summary.items():
        log.info("%s %s", st, v)
    return {"rows": rows, "summary": summary}


def say_table(words):
    """Every (state, cue) -> word the speaker needs for `S li P [e O]` / `mi|sina P [e O]` replies.
    A cue is the next slot word tagged with its slot; "-" = nothing left (end)."""
    from chatfly.toki import END

    t = []
    for w in words:
        t += [("S0", ("subj", w), w), ("SUBJ", ("subj", w), w), ("SUBJ", ("pred", w), "li"),
              ("LI", ("pred", w), w), ("MI", ("pred", w), w), ("PRED", ("pred", w), w), ("MIP", ("pred", w), w),
              ("PRED", ("obj", w), "e"), ("MIP", ("obj", w), "e"), ("E", ("obj", w), w), ("OBJ", ("obj", w), w)]
    t += [(s, ("-", "-"), END) for s in ("PRED", "MIP", "OBJ")]
    return t


def _sjob(args):
    from chatfly.toki import END, GRAMMAR, NAME, klass, slots

    setting, cfg, seed = args
    sp, t0 = cfg["speaker"], time.time()
    if not CLAUSES:
        CLAUSES.extend(tatoeba_clauses())
    words = CONTENT + [NAME]
    f = {**sp["settings"][setting.removesuffix("_no_kc")], "n_pn": cfg["mb"]["n_pn"], "code": tuple(sp["code"])}
    od = Odors(cfg, seed + 700, cfg["mb"]["n_pn"], cfg["tokens"]["pn_active_frac"])
    fly = Fly(cfg, seed, 4, words + ["li", "e", END], f, "no_kc" if setting.endswith("_no_kc") else None)
    tab = [(s, c, o) for s, c, o in say_table(words) if not (s == "MI" and c[1] in ("mi", "sina"))]
    def x(s, c):
        return np.maximum(od("s", s), od(*c))
    rng = np.random.default_rng([seed, 9, 5])
    # balance (8a-3): the rare end contexts as often as a word context of their state
    order = np.repeat(np.arange(len(tab)), [sp["end_reps"] if o == END else 1 for _, _, o in tab])
    for _ in range(sp["epochs"]):
        for i in rng.permutation(order):
            fly.teach(x(*tab[i][:2]), tab[i][2])
    for _ in range(sp["rehearse"]):
        for i in rng.permutation(len(tab)):
            if fly.read(x(*tab[i][:2]))[0] != tab[i][2]:
                fly.teach(x(*tab[i][:2]), tab[i][2])
    step = float(np.mean([fly.read(x(s, c))[0] == o for s, c, o in tab]))
    def speak(s_, p_, o_):  # the cue pointer moves on when the cue word is said (8h: a fly)
        cues = [("subj", w) for w in s_] + [("pred", w) for w in p_] + [("obj", w) for w in o_]
        state, k, out = "S0", 0, []
        while len(out) < 16:
            w = fly.read(x(state, cues[k] if k < len(cues) else ("-", "-")))[0]
            if w is None or w == END:
                break
            out.append(w)
            state = GRAMMAR[state].get(klass(w), "X")
            k += k < len(cues) and cues[k][1] == w
        return out
    facts = [f for f in (slots(CLAUSES[i]) for i in rng.permutation(len(CLAUSES))[:sp["n_test"] * 3]) if f][:sp["n_test"]]
    def truth(s_, p_, o_):
        bare = len(s_) == 1 and s_[0] in ("mi", "sina")
        return list(s_) + ([] if bare else ["li"]) + list(p_) + (["e", *o_] if o_ else [])
    sent = float(np.mean([speak(*f[:3]) == truth(*f[:3]) for f in facts]))
    out = {"setting": setting, "seed": seed, "contexts": len(tab), "table": step, "tatoeba_sentences": sent, "s": round(time.time() - t0, 1)}
    log.info("%s", out)
    return out


def exp_probe9_speaker(cfg: dict, seed: int) -> dict:
    cfg = with_base(cfg)
    sp = cfg["speaker"]
    jobs = [(st, cfg, s) for st in sp["run"] for s in cfg["select_seeds"]]
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        rows = list(ex.map(_sjob, jobs))
    summary = {st: {k: [round(float(np.mean([r[k] for r in rows if r["setting"] == st])), 4),
                        round(float(np.min([r[k] for r in rows if r["setting"] == st])), 4)]
                    for k in ("table", "tatoeba_sentences")} for st in sp["run"]}
    for st, v in summary.items():
        log.info("%s %s", st, v)
    return {"rows": rows, "summary": summary}
