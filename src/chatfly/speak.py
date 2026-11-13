"""Phase 7a: a Zumi dialogue corpus, generation metrics and non-fly baselines (ADR 0012).

Phase 7 goal: a fly that *says* its Zumi replies word by word. 7a has no fly. It fixes
- the corpus: dialogues of (user utterance, correct fly reply), generated from the Zumi
  grammar and a hidden world (ring place, taught facts) that lives only here, in the oracle;
- the metrics: teacher-forced reply perplexity / next-word accuracy, and free generation
  scored for grammaticality (reply grammar), appropriateness (reply form fits the question)
  and exactness (= the oracle's reply);
- the baselines the fly must be compared with: n-grams (nouns as one class) and a small
  Elman RNN that sees each noun as a fixed random vector, the same information the fly gets
  (a noun's PN pattern). Facts and the ring place change within a dialogue, so exact replies
  need memory across turns: n-grams can't have it; the RNN must learn it in context.

Stream per turn: user words, SEP, reply words, EOS. Replies never contain nouns.
"""

import hashlib
import logging

import numpy as np

from chatfly.fsm import make_task
from chatfly.zumi import parse, props, random_utterance, reserved

log = logging.getLogger("chatfly")
SEP, EOS, NOUN = "|", "#", "N"
KINDS = ("teach", "ask", "what", "ring", "bad")


# --- corpus -------------------------------------------------------------------------------

def kind_of(p) -> str:
    """Utterance type used by the metrics."""
    if "refuse" in p:
        return "bad"
    if "noun" in p:
        return p["kind"]
    return {None: "statement", "where": "where"}.get(p["ask"], "yesno")


def oracle(p, world, task, lx) -> list:
    """The correct reply; updates world {"place", "facts": {noun: {pair: side}}}. The fly
    never sees this function (evaluation and training targets only)."""
    r, pr = lx["replies"], props(lx)
    if "refuse" in p:
        return [r["huh"]]
    if "noun" in p:
        f = world["facts"].setdefault(p["noun"], {})
        if p["kind"] == "teach":
            for w in p["props"]:
                f[pr[w][0]] = pr[w][1]
            return [r["ok"]]
        if p["kind"] == "what":
            return [list(lx["props"][i])[f[i]] for i in sorted(f)] or [r["unsure"]]
        i, side = pr[p["props"][0]]
        return [r["unsure"]] if i not in f else [r["yes"] if f[i] == side else r["no"]]
    s = task["states"][0] if p["restart"] else world["place"]
    for m in p["moves"]:
        s = task["delta"][s][m]
    if not p["suppose"]:
        world["place"] = s
    if p["ask"] is None:
        return [r["ok"]]
    return [s] if p["ask"] == "where" else [r["yes"] if s == p["ask"] else r["no"]]


def utterance(rng, lx, nouns, co) -> list:
    """One user utterance (list of words) of a random kind; "bad" ones are refused by parse."""
    kind = KINDS[rng.choice(len(KINDS), p=co["p_kind"])]
    pairs = lx["props"]
    if kind == "ring":
        w = random_utterance(lx, rng, co["ring"]).rstrip(".?").split()
        return [lx["restart"]] + w if rng.random() < co["p_restart"] else w
    noun = str(rng.choice(nouns))
    if kind == "teach":
        k = rng.integers(1, co["max_teach"] + 1)
        return [noun, lx["is"]] + [list(pairs[i])[rng.integers(2)]
                                   for i in rng.choice(len(pairs), k, replace=False)]
    if kind == "ask":
        return [noun, list(pairs[rng.integers(len(pairs))])[rng.integers(2)], lx["yesno"]]
    if kind == "what":
        return [noun, lx["what"]]
    vocab = sorted(reserved(lx) - set(lx["replies"].values())) + list(nouns)
    while True:  # bad: random word salad that the grammar refuses
        w = [str(x) for x in rng.choice(vocab, rng.integers(1, 5))]
        if "refuse" in parse(" ".join(w), lx):
            return w


def noun_word(rng, co, banned) -> str:
    lo, hi = co["noun_len"]
    while True:
        w = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), rng.integers(lo, hi + 1)))
        if w not in banned:
            return w


def dialogue(rng, lx, task, co) -> list:
    """[(user words, reply words, kind)] for one conversation with a fresh world."""
    banned = reserved(lx)
    nouns = [noun_word(rng, co, banned) for _ in range(co["n_nouns"])]
    world, out = {"place": task["states"][0], "facts": {}}, []
    for _ in range(co["turns"]):
        u = utterance(rng, lx, nouns, co)
        p = parse(" ".join(u), lx)
        out.append((u, oracle(p, world, task, lx), kind_of(p)))
    return out


def corpus(cfg, seed, split) -> list:
    co = cfg["corpus"]
    rng = np.random.default_rng([seed, cfg["rng_streams"]["corpus"], ("train", "dev", "test").index(split)])
    task = make_task(cfg["chat"]["task"], None)
    return [dialogue(rng, cfg["lang"], task, co) for _ in range(co["n_" + split])]


# --- metrics ------------------------------------------------------------------------------

def grammatical(reply, lx) -> bool:
    """Reply grammar: one of o ya na hm e <place>, or properties one per pair in pair order."""
    r, pr = lx["replies"], props(lx)
    if len(reply) == 1 and (reply[0] in r.values() or reply[0] in lx["places"]):
        return True
    idx = [pr[w][0] if w in pr else None for w in reply]
    return bool(reply) and None not in idx and idx == sorted(set(idx))


def appropriate(kind, reply, lx) -> bool:
    """The reply has a form that fits the utterance (content may still be wrong)."""
    r, one = lx["replies"], reply[0] if len(reply) == 1 else None
    yn = {r["yes"], r["no"], r["unsure"]}
    return {"teach": one == r["ok"], "ask": one in yn, "yesno": one in yn, "bad": one == r["huh"],
            "statement": one in (r["ok"], r["unsure"]),
            "where": one in lx["places"] or one == r["unsure"],
            "what": one == r["unsure"] or (grammatical(reply, lx) and reply[0] in props(lx)),
            }[kind]


def evaluate(model, dialogues, lx, max_len=6) -> dict:
    """Teacher-forced reply NLL / next-word accuracy, and greedy free generation of each
    reply from the gold history."""
    nll, hit, n_tok, rows = 0.0, 0, 0, []
    for d in dialogues:
        st = model.init()
        for user, reply, kind in d:
            for t in user + [SEP]:
                st, dist = model.step(st, t)
            gen, s, g = [], st, dist
            for _ in range(max_len):
                t = model.vocab[int(np.argmax(g))]
                if t == EOS:
                    break
                gen.append(t)
                s, g = model.step(s, t)
            for t in reply + [EOS]:
                p = dist[model.index[t]]
                nll -= np.log(max(p, 1e-12))
                hit += int(np.argmax(dist)) == model.index[t]
                n_tok += 1
                st, dist = model.step(st, t)
            rows.append((kind, gen == reply, grammatical(gen, lx), bool(gen) and appropriate(kind, gen, lx)))
    out = {"perplexity": float(np.exp(nll / n_tok)), "next_word_acc": hit / n_tok}
    for name, j in (("exact", 1), ("grammatical", 2), ("appropriate", 3)):
        out[name] = float(np.mean([x[j] for x in rows]))
    out["exact_by_kind"] = {k: float(np.mean([x[1] for x in rows if x[0] == k]))
                            for k in sorted({x[0] for x in rows})}
    return out


def vocabulary(lx) -> list:
    return sorted(reserved(lx)) + [NOUN, SEP, EOS]


# --- baselines ----------------------------------------------------------------------------

class Oracle:
    """Sanity check: the metrics must give 1.0 for the oracle."""

    def __init__(self, cfg):
        self.lx, self.task = cfg["lang"], make_task(cfg["chat"]["task"], None)
        self.vocab = vocabulary(self.lx)
        self.index = {w: i for i, w in enumerate(self.vocab)}

    def init(self):
        return ({"place": self.task["states"][0], "facts": {}}, [], [])

    def step(self, st, t):
        world, user, todo = st[0], list(st[1]), list(st[2])
        if t == SEP:
            world = {"place": world["place"], "facts": {k: dict(v) for k, v in world["facts"].items()}}
            todo, user = oracle(parse(" ".join(user), self.lx), world, self.task, self.lx) + [EOS], []
        elif todo:
            todo.pop(0)
        else:
            user.append(t)
        dist = np.zeros(len(self.vocab))
        dist[self.index[todo[0] if todo else EOS]] = 1.0
        return (world, user, todo), dist


class NGram:
    """Backoff to the longest seen context (nouns mapped to N), add-alpha at that order."""

    def __init__(self, n, alpha, vocab):
        self.n, self.alpha, self.vocab = n, alpha, vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        self.counts = {}

    def tok(self, t):
        return t if t in self.index else NOUN

    def fit(self, dialogues):
        for d in dialogues:
            h = ()
            for user, reply, _ in d:
                for t in user + [SEP] + reply + [EOS]:
                    t = self.tok(t)
                    for k in range(len(h) + 1):
                        c = self.counts.setdefault(h[len(h) - k:], np.zeros(len(self.vocab)))
                        c[self.index[t]] += 1
                    h = (h + (t,))[-(self.n - 1):] if self.n > 1 else ()
        return self

    def init(self):
        return ()

    def step(self, h, t):
        h = (h + (self.tok(t),))[-(self.n - 1):] if self.n > 1 else ()
        for k in range(len(h), -1, -1):
            c = self.counts.get(h[len(h) - k:])
            if c is not None:
                return h, (c + self.alpha) / (c.sum() + self.alpha * len(c))
        return h, np.full(len(self.vocab), 1 / len(self.vocab))


class RNN:
    """Elman RNN (tanh), trained with full BPTT per dialogue on reply-word targets only.
    Input: one-hot word (nouns -> N) plus, for a noun, a fixed random +-1 vector from
    sha256(word), like the fly's PN pattern for a noun."""

    def __init__(self, vocab, rc, seed):
        self.vocab, self.rc = vocab, rc
        self.index = {w: i for i, w in enumerate(vocab)}
        rng = np.random.default_rng(seed)
        v, d, h = len(vocab), rc["noun_dim"], rc["hidden"]
        s = lambda *shape: rng.normal(0, 1 / np.sqrt(shape[0]), shape)
        self.p = {"Wx": s(v + d, h), "Wh": s(h, h) * 0.5, "b": np.zeros(h),
                  "Wo": s(h, v), "bo": np.zeros(v)}

    def x(self, t):
        v, d = len(self.vocab), self.rc["noun_dim"]
        out = np.zeros(v + d)
        if t in self.index:
            out[self.index[t]] = 1
        else:
            out[self.index[NOUN]] = 1
            r = np.random.default_rng(int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little"))
            out[v:] = r.choice([-1.0, 1.0], d) / np.sqrt(d)
        return out

    def init(self):
        return np.zeros(self.rc["hidden"])

    def step(self, h, t):
        p = self.p
        h = np.tanh(self.x(t) @ p["Wx"] + h @ p["Wh"] + p["b"])
        z = h @ p["Wo"] + p["bo"]
        e = np.exp(z - z.max())
        return h, e / e.sum()

    def batch(self, ds):
        """Inputs X (T, B, in), targets Y (T, B) with -1 where there is no loss."""
        seqs = []
        for d in ds:
            xs, ys = [], []
            for user, reply, _ in d:
                toks = user + [SEP] + reply + [EOS]
                for i, t in enumerate(toks[:-1]):
                    xs.append(self.x(t))
                    ys.append(self.index[toks[i + 1]] if i >= len(user) else -1)
                xs.append(self.x(EOS))
                ys.append(-1)
            seqs.append((np.array(xs), np.array(ys)))
        T = max(len(y) for _, y in seqs)
        X = np.zeros((T, len(ds), seqs[0][0].shape[1]))
        Y = -np.ones((T, len(ds)), dtype=int)
        for b, (x, y) in enumerate(seqs):
            X[:len(y), b], Y[:len(y), b] = x, y
        return X, Y

    def loss_grad(self, X, Y):
        p, (T, B, _) = self.p, X.shape
        H = np.zeros((T + 1, B, self.rc["hidden"]))
        P = np.zeros((T, B, len(self.vocab)))
        for t in range(T):
            H[t + 1] = np.tanh(X[t] @ p["Wx"] + H[t] @ p["Wh"] + p["b"])
            z = H[t + 1] @ p["Wo"] + p["bo"]
            e = np.exp(z - z.max(1, keepdims=True))
            P[t] = e / e.sum(1, keepdims=True)
        m = Y >= 0
        n = max(m.sum(), 1)
        loss = -np.log(P[m, Y[m]] + 1e-12).sum() / n
        dZ = P.copy()
        dZ[m, Y[m]] -= 1
        dZ *= m[..., None] / n
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dh = np.zeros((B, self.rc["hidden"]))
        for t in range(T - 1, -1, -1):
            g["Wo"] += H[t + 1].T @ dZ[t]
            g["bo"] += dZ[t].sum(0)
            da = (dZ[t] @ p["Wo"].T + dh) * (1 - H[t + 1] ** 2)
            g["Wx"] += X[t].T @ da
            g["Wh"] += H[t].T @ da
            g["b"] += da.sum(0)
            dh = da @ p["Wh"].T
        return loss, g

    def fit(self, train, dev, lx, seed):
        """Adam; keeps the epoch with the lowest dev perplexity."""
        rc, rng = self.rc, np.random.default_rng(seed + 1)
        mom = {k: np.zeros_like(v) for k, v in self.p.items()}
        vel = {k: np.zeros_like(v) for k, v in self.p.items()}
        best, best_p, step, hist = np.inf, None, 0, []
        for ep in range(rc["epochs"]):
            order = rng.permutation(len(train))
            for i in range(0, len(train), rc["batch"]):
                _, g = self.loss_grad(*self.batch([train[j] for j in order[i:i + rc["batch"]]]))
                norm = np.sqrt(sum((v ** 2).sum() for v in g.values()))
                step += 1
                for k in self.p:
                    gk = g[k] * min(1.0, rc["clip"] / (norm + 1e-12))
                    mom[k] = 0.9 * mom[k] + 0.1 * gk
                    vel[k] = 0.999 * vel[k] + 0.001 * gk ** 2
                    self.p[k] -= rc["lr"] * (mom[k] / (1 - 0.9 ** step)) / (
                        np.sqrt(vel[k] / (1 - 0.999 ** step)) + 1e-8)
            ppl = evaluate(self, dev, lx)["perplexity"]
            hist.append(ppl)
            log.info("rnn seed %d epoch %d dev perplexity %.4f", seed, ep, ppl)
            if ppl < best:
                best, best_p = ppl, {k: v.copy() for k, v in self.p.items()}
        self.p = best_p
        return hist


def exp_speak_baselines(cfg: dict, seed: int) -> dict:
    """Corpus checks, then every baseline on each seed's test corpus (ADR 0012)."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    cfg = {**yaml.safe_load((root / cfg["base"]).read_text()), **cfg}
    lx, bc, out = cfg["lang"], cfg["baselines"], []
    for s in cfg["seeds"]:
        train, dev, test = (corpus(cfg, s, k) for k in ("train", "dev", "test"))
        kinds = [k for d in test for _, _, k in d]
        row = {"seed": s, "kind_freq": {k: kinds.count(k) / len(kinds) for k in sorted(set(kinds))},
               "sample": [[" ".join(u), " ".join(r)] for u, r, _ in test[0][:12]], "models": {}}
        vocab = vocabulary(lx)
        models = {"oracle": Oracle(cfg)}
        for n in bc["ngram_orders"]:
            models[f"{n}-gram"] = NGram(n, bc["alpha"], vocab).fit(train)
        rnn = RNN(vocab, bc["rnn"], s)
        row["rnn_dev_perplexity"] = rnn.fit(train, dev, lx, s)
        models["rnn"] = rnn
        for name, m in models.items():
            row["models"][name] = r = evaluate(m, test, lx)
            log.info("seed %d %-8s ppl %.3f next %.3f exact %.3f gram %.3f approp %.3f by kind %s",
                     s, name, r["perplexity"], r["next_word_acc"], r["exact"], r["grammatical"],
                     r["appropriate"], {k: round(v, 3) for k, v in r["exact_by_kind"].items()})
        out.append(row)
    summary = {}
    for name in out[0]["models"]:
        rs = [r["models"][name] for r in out]
        summary[name] = {k: [float(np.mean([r[k] for r in rs])), float(np.min([r[k] for r in rs]))]
                         for k in ("perplexity", "next_word_acc", "exact", "grammatical", "appropriate")}
        summary[name]["exact_by_kind"] = {k: float(np.mean([r["exact_by_kind"][k] for r in rs]))
                                          for k in rs[0]["exact_by_kind"]}
        log.info("summary %-8s %s", name, summary[name])
    return {"summary": summary, "seeds": out}
