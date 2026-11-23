"""Phase 9 (ADR 0014): the Toki Pona chatbot.

One `Agent` runs a conversation from three parts:
- a parser (words -> grammar states);
- a memory (key parts -> answer phrase);
- a speaker (reply slots -> words).

The reference `oracle()` uses the textbook grammar, a dict and `render`. The fly `bot()` uses the
grammar fly, the chained-key memory (binder + recall fly) and the speaker fly. Everything else
is the same code. That code holds the interface rules (glue, counted in ADR 0014): roles from
grammar states, the mi/sina swap, the question-form and act rules, the keys a fact is stored
under, the reply plan, and the cue pointer.

`corpus()` makes conversations from real Tatoeba clauses with the acceptable replies. They come
from the facts the generator chose, not from parsing its own text.
"""

import logging
from pathlib import Path

import numpy as np
import yaml

from chatfly.toki import (
    NAME,
    ROLE,
    parse,
    tokens,
)

log = logging.getLogger("chatfly")
ROOT = Path(__file__).resolve().parents[2]
SWAP = {"mi": "sina", "sina": "mi"}
SLOTS = ("ctx", "subj", "pred", "obj", "prep", "pobj")
SOCIAL = {  # act -> (user phrases, the reply); conventions, not facts
    "greet": ({"toki", "toki a", "a toki"}, "toki"),
    "thanks": ({"pona", "pona a", "pona tawa sina"}, "pona tawa sina"),
    "farewell": ({"mi tawa", "tawa pona", "mi tawa a"}, "tawa pona"),
}
REPLY = {"command": "mi ken ala", "not_understood": "mi sona ala e toki sina", "unknown": "mi sona ala",
         "new": "pona", "known": "mi sona"}
# the bot's view; "sina seme?" gets the latest predicate told about mi, so the core one is last
PERSONA = ["mi moku ala", "mi lon ilo", "mi pilin pona", "nimi mi li pipi", "mi pipi lili"]


# ---- slots, keys and replies (shared by the oracle and the bot) ----
def slots_of(words, states):
    """Grammar states -> {slot: words} for one clause (a dropped, mi/sina swapped), or None if
    the clause has more than one li, e, prepositional phrase or la (not stored, not asked)."""
    r = [ROLE[s] for s in states]
    if r.count("li") > 1 or r.count("e") > 1 or r.count("prep") > 1 or r.count("la") > 1:
        return None
    if "la" in r:
        i = r.index("la")
        ctx, words, r = tuple(words[:i]), words[i + 1:], r[i + 1:]
    else:
        ctx = ()
    if not words:
        return None
    if words[0] in SWAP and "li" in r:  # "mi mute li ...": the words before li are the subject
        j = r.index("li")
        r = ["subj"] * j + r[j:]
    out = {"ctx": ctx, **{k: tuple(w for w, x in zip(words, r) if x == k and w != "a") for k in SLOTS[1:]}}
    if "o" in r:
        out["o"] = True
    return {k: tuple(SWAP.get(w, w) for w in v) if isinstance(v, tuple) else v for k, v in out.items()}


def render(s):
    """The reference sentence for reply slots."""
    bare = s.get("subj", ()) in (("mi",), ("sina",))
    out = list(s.get("ctx", ())) + (["la"] if s.get("ctx") else []) + list(s.get("subj", ()))
    out += ([] if bare or not s.get("pred") else ["li"]) + list(s.get("pred", ()))
    out += (["e", *s["obj"]] if s.get("obj") else []) + list(s.get("prep", ())) + list(s.get("pobj", ()))
    return out


def cues(s):
    """Reply slots -> the speaker's cue list (slot-tagged words, in render order)."""
    return [(k, w) for k in SLOTS for w in s.get(k, ())]


def keys(f):
    """A told fact {ctx, subj, pred, obj, prep, pobj, neg} -> [(form, key parts, answer)]."""
    S, P, O, neg = f["subj"], f["pred"], f["obj"], f["neg"]
    yn = ("ala",) if neg else ("lon",)
    out = [("yn", (("subj", S), ("pred", P), ("q", ("anu",))), yn),
           ("pred", (("subj", S), ("q", ("li",))), P + (("ala",) if neg else ()))]
    if O:
        out += [("yn", (("subj", S), ("pred", P), ("obj", O), ("q", ("anu",))), yn),
                ("obj", (("subj", S), ("pred", P), ("q", ("e",))), O)]
    if not neg:
        out.append(("subj", (("q", ("seme",)), ("pred", P), ("obj", O or ("-",))), S))
        if len(P) > 1:
            out.append(("pmod", (("subj", S), ("pred", P[:1]), ("q", ("seme",))), P[1:]))
    if f["prep"]:
        out.append(("prep", (("subj", S), ("pred", P), ("prep", f["prep"])), f["pobj"]))
    if f["ctx"]:
        out.append(("ctx", (("subj", S), ("pred", P), ("q", ("la",))), f["ctx"]))
    return out


def as_fact(s):
    """Slots of a statement -> a fact, or None if it cannot be stored."""
    if not s or s.get("o") or not s["subj"] or not s["pred"] or "seme" in sum((s[k] for k in SLOTS), ()):
        return None
    if s["ctx"] and s["ctx"][0] != "tenpo":  # only time contexts are asked about ("tenpo seme la")
        return None
    P = s["pred"]
    if "ala" in P[1:] and P.index("ala", 1) == 1 and len(P) > 2 and P[2] == P[0]:
        return None  # "P ala P" is a question form
    neg = P[-1] == "ala" and len(P) > 1
    P = P[:-1] if neg else P
    if "ala" in P or "anu" in sum((s[k] for k in SLOTS), ()) or "en" in sum((s[k] for k in SLOTS), ()):
        return None
    return {**{k: s[k] for k in SLOTS}, "pred": P, "neg": neg}


def is_question(words, s):
    P = s["pred"]
    return "seme" in words or (len(P) >= 3 and P[1] == "ala" and P[2] == P[0])


def question(s):
    """Slots of a question -> (form, key parts, reply slots with None where the answer goes), or None."""
    if s is None:
        return None
    s = {k: s[k] for k in SLOTS} | ({"o": True} if s.get("o") else {})
    tail = None
    for k in reversed(SLOTS):  # "... anu seme": a yes/no question
        if s[k][-2:] == ("anu", "seme"):
            s[k], tail = s[k][:-2], "yn"
            break
        if s[k]:
            break
    P = s["pred"]
    if tail is None and len(P) >= 3 and P[1] == "ala" and P[2] == P[0]:  # "P ala P"
        s["pred"], tail = (P[0],) + P[3:], "yn"
    S, P, O = s["subj"], s["pred"], s["obj"]
    if tail == "yn":
        if not S or not P or "seme" in S + P + O:
            return None
        parts = (("subj", S), ("pred", P)) + ((("obj", O),) if O else ()) + (("q", ("anu",)),)
        return "yn", parts, s
    if S == ("seme",) and P:
        return "subj", (("q", ("seme",)), ("pred", P), ("obj", O or ("-",))), {**s, "subj": None}
    if not S or "seme" in S:
        return None
    if O == ("seme",):
        return "obj", (("subj", S), ("pred", P), ("q", ("e",))), {**s, "obj": None}
    if P == ("seme",) and not O:
        return "pred", (("subj", S), ("q", ("li",))), {**s, "pred": None}
    if len(P) > 1 and P[-1] == "seme" and "seme" not in P[:-1] and not O:
        return "pmod", (("subj", S), ("pred", P[:-1]), ("q", ("seme",))), {**s, "pred": P[:-1] + (None,)}
    if s["pobj"] == ("seme",) and s["prep"]:
        return "prep", (("subj", S), ("pred", P), ("prep", s["prep"])), {**s, "pobj": None}
    if s["ctx"] == ("tenpo", "seme"):
        return "ctx", (("subj", S), ("pred", P), ("q", ("la",))), {**s, "ctx": None}
    return None


def fill(form, plan, ans):
    """Put the answer phrase in the reply slots."""
    if form == "yn":
        return {**plan, "pred": plan["pred"] + (("ala",) if ans == ("ala",) else ())}
    if form == "pmod":
        return {**plan, "pred": plan["pred"][:-1] + ans}
    slot = {"subj": "subj", "obj": "obj", "pred": "pred", "prep": "pobj", "ctx": "ctx"}[form]
    return {**plan, slot: ans}


def social_slots(text):
    w = text.split()
    st = parse(w)
    return {k: tuple(v) for k, v in zip(SLOTS, ((),) * 6)} | (_raw(w, st) if st else {})


def _raw(words, states):
    """Slots without the swap (for the bot's own fixed replies)."""
    s = slots_of(words, states)
    return {k: tuple(SWAP.get(x, x) for x in v) if isinstance(v, tuple) else v for k, v in s.items()}


class Agent:
    """parse: words -> states or None; mem: .tell(parts, phrase) / .ask(parts) -> phrase or None;
    say: reply slots -> words."""

    def __init__(self, parse_fn, mem, say_fn):
        self.parse, self.mem, self.say = parse_fn, mem, say_fn
        self.fixed = {act: social_slots(r) for act, r in REPLY.items()} | {a: social_slots(r) for a, (_, r) in SOCIAL.items()}

    def tell(self, fact):
        for _, parts, ans in keys(fact):
            self.mem.tell(parts, ans)

    def act(self, text):
        """-> (act, detail) per clause: the act rules (9c replaces them with flies)."""
        out = []
        for words in tokens(text):
            if words is None or NAME in words:
                out.append(("not_understood", None))
                continue
            social = [a for a, (ph, _) in SOCIAL.items() if " ".join(words) in ph]
            if social:
                out.append((social[0], None))
                continue
            st = self.parse(words)
            s = slots_of(words, st) if st else None
            if s is None:
                out.append(("not_understood", None))
            elif s.get("o"):
                out.append(("command", None))
            elif is_question(words, s):
                q = question(s)
                out.append(("question", q) if q else ("not_understood", None))
            else:
                f = as_fact(s)
                out.append(("statement", f) if f else ("not_understood", None))
        return out

    def reply(self, text):
        said = []
        for act, d in self.act(text):
            if act == "question":
                form, parts, plan = d
                ans = self.mem.ask(parts)
                said.append(self.say(self.fixed["unknown"] if ans is None else fill(form, plan, ans)))
            elif act == "statement":
                same = self.mem.ask(keys(d)[0][1]) == keys(d)[0][2]
                self.tell(d)
                said.append(self.say(self.fixed["known" if same else "new"]))
            else:
                said.append(self.say(self.fixed[act]))
        return ". ".join(" ".join(w) for w in said)


class DictMemory:
    def __init__(self):
        self.d = {}

    def tell(self, parts, ans):
        self.d[parts] = ans

    def ask(self, parts):
        return self.d.get(parts)


def oracle():
    a = Agent(parse, DictMemory(), render)
    for p in PERSONA:
        a.tell(_persona(p))
    return a


def _persona(p):
    """A persona sentence is in the bot's own view: no swap."""
    return as_fact(_raw(p.split(), parse(p.split())))


# ---- the corpus ----
def tatoeba_facts():
    """Facts from real clauses (reference parse, user's view = the bot's view swapped)."""
    path = ROOT / "data/tatoeba/tok_sentences.tsv"
    out, seen = [], set()
    for line in path.read_text(encoding="utf8").splitlines():
        for w in tokens(line.split("\t")[2]):
            if not w or NAME in w or len(w) > 12 or " ".join(w) in set().union(*(ph for ph, _ in SOCIAL.values())):
                continue
            st = parse(w)
            f = as_fact(slots_of(w, st)) if st else None
            if f and " ".join(w) not in seen and _user_text(f) == w and all(_round_trip(x, f) for x in forms_of(f)):
                seen.add(" ".join(w))
                out.append(f)
    return out


def _round_trip(form, f):
    """The question text of a form parses back to that form (e.g. "tawa ala tawa" does not: the
    grammar reads the second tawa as a preposition)."""
    t = _q_text(form, f)
    st = parse(t)
    q = question(slots_of(t, st)) if st else None
    return q is not None and q[0] == form.removesuffix("_ala")


def _user_text(f):
    """What the user says to tell fact f (their view)."""
    sw = {k: tuple(SWAP.get(x, x) for x in v) for k, v in f.items() if k in SLOTS}
    return render({**sw, "pred": sw["pred"] + (("ala",) if f["neg"] else ())})


def _q_text(form, f):
    """What the user says to ask form about fact f."""
    sw = {k: tuple(SWAP.get(x, x) for x in v) for k, v in f.items() if k in SLOTS}
    S, P, O = sw["subj"], sw["pred"], sw["obj"]
    if form == "yn":
        return render({"subj": S, "pred": P, "obj": O}) + ["anu", "seme"]
    if form == "yn_ala":
        return render({"subj": S, "pred": (P[0], "ala") + P, "obj": O})
    if form == "obj":
        return render({"subj": S, "pred": P, "obj": ("seme",)})
    if form == "pred":
        return render({"subj": S, "pred": ("seme",)})
    if form == "pmod":
        return render({"subj": S, "pred": P[:1] + ("seme",)})
    if form == "subj":
        return render({"subj": ("seme",), "pred": P, "obj": O})
    if form == "prep":
        return render({"subj": S, "pred": P, "obj": O, "prep": sw["prep"], "pobj": ("seme",)})
    return render({"ctx": ("tenpo", "seme"), "subj": S, "pred": P, "obj": O})


def forms_of(f):
    out = ["yn", "yn_ala", "pred"] + (["obj"] if f["obj"] else []) + (["subj"] if not f["neg"] else [])
    out += (["pmod"] if len(f["pred"]) > 1 and not f["neg"] and not f["obj"] else []) + (["prep"] if f["prep"] else [])
    out += ["ctx"] if f["ctx"] else []
    # yn_ala repeats the first predicate word: only when that word is not a preposition-like head
    return [x for x in out if x != "yn_ala" or (len(f["pred"]) >= 1 and f["pred"][0] not in ("ala",))]


def corpus(cfg, seed, facts):
    """One conversation: [(user text, set of acceptable replies)]. The world model: the same
    keys rule (latest wins) over the facts the generator chose; subject questions accept every
    subject told with that (P, O)."""
    c = cfg["corpus"]
    rng = np.random.default_rng([seed, 9, 30])
    pool = [facts[i] for i in rng.permutation(len(facts))[:c["n_pool"]]]
    told_pool, unknown_pool = pool[:c["n_pool"] // 2], pool[c["n_pool"] // 2:]
    world, subjects, told = {}, {}, []
    def tell(f):
        for _, parts, ans in keys(f):
            world[parts] = ans
            if parts[0] == ("q", ("seme",)):
                subjects.setdefault(parts, set()).add(ans)
    for p in PERSONA:
        f = _persona(p)
        tell(f)
        told.append(f)
    turns = []
    acts = list(c["p_turn"])
    for _ in range(c["n_turns"]):
        a = acts[int(rng.choice(len(acts), p=list(c["p_turn"].values())))]
        if a == "tell" or (a in ("retell", "change", "ask") and len(told) <= len(PERSONA)):
            f = told_pool[len(told) % len(told_pool)]
            same = world.get(keys(f)[0][1]) == keys(f)[0][2]
            turns.append((" ".join(_user_text(f)), {REPLY["known" if same else "new"]}))
            tell(f)
            told.append(f)
        elif a == "retell":
            f = told[int(rng.integers(len(told)))]
            if f in [_persona(p) for p in PERSONA]:
                continue
            same = world.get(keys(f)[0][1]) == keys(f)[0][2]
            turns.append((" ".join(_user_text(f)), {REPLY["known" if same else "new"]}))
            tell(f)
        elif a == "change":  # a change of mind: the other polarity
            f = told[int(rng.integers(len(PERSONA), len(told)))]
            g = {**f, "neg": not f["neg"]}
            same = world.get(keys(g)[0][1]) == keys(g)[0][2]
            turns.append((" ".join(_user_text(g)), {REPLY["known" if same else "new"]}))
            tell(g)
            told.append(g)
        elif a in ("ask", "ask_unknown"):
            f = told[int(rng.integers(len(told)))] if a == "ask" else unknown_pool[int(rng.integers(len(unknown_pool)))]
            fs = forms_of(f)
            form = fs[int(rng.integers(len(fs)))]
            text = _q_text(form, f)
            if not _round_trip(form, f):  # a changed-mind or persona fact may not have every form
                continue
            kind, parts, plan = question(slots_of(text, parse(text)))
            if parts[0] == ("q", ("seme",)):
                ok = {" ".join(render(fill(kind, plan, s))) for s in subjects.get(parts, ())}
            else:
                ok = {" ".join(render(fill(kind, plan, world[parts])))} if parts in world else set()
            turns.append((" ".join(text), ok or {REPLY["unknown"]}))
        elif a == "social":
            act = ["greet", "thanks", "farewell", "command"][int(rng.integers(4))]
            if act == "command":
                f = told_pool[int(rng.integers(len(told_pool)))]
                sw = {k: tuple(SWAP.get(x, x) for x in f[k]) for k in SLOTS}
                text = " ".join(["o", *sw["pred"]] + (["e", *sw["obj"]] if sw["obj"] else []) + list(sw["prep"] + sw["pobj"]))
                turns.append((text, {REPLY["command"]}))
            else:
                ph = sorted(SOCIAL[act][0])
                turns.append((ph[int(rng.integers(len(ph)))], {SOCIAL[act][1]}))
        else:  # not understood: a word salad, or a clause with a word outside nimi pu
            f = told_pool[int(rng.integers(len(told_pool)))]
            w = list(_user_text(f))
            text = " ".join(rng.permutation(w)) if rng.random() < 0.5 else " ".join(w[:1] + ["kijetesantakalu"] + w[1:])
            if parse(text.split()) is not None and "kijetesantakalu" not in text:
                continue
            turns.append((text, {REPLY["not_understood"]}))
    return turns


def exp_corpus9(cfg: dict, seed: int) -> dict:
    """9a: the oracle on every conversation must score 1.0."""
    cfg = {**yaml.safe_load((ROOT / cfg["base"]).read_text()), **cfg}
    facts = tatoeba_facts()
    rows = []
    for s in cfg["seeds"]:
        turns = corpus(cfg, s, facts)
        a = oracle()
        wrong, ok = [], 0
        for t, good in turns:
            r = a.reply(t)
            ok += r in good
            if r not in good and len(wrong) < 5:
                wrong.append((t, r, sorted(good)))
        rows.append({"seed": s, "turns": len(turns), "oracle": ok / len(turns), "wrong": wrong})
        log.info("%s", rows[-1])
    return {"n_facts": len(facts), "rows": rows, "oracle_min": min(r["oracle"] for r in rows)}
