# Zumi — a guidebook

Zumi is a tiny language for talking with the digital fly (a connectome-style computer model
of the fly mushroom body, not a living fly). You can talk about **anything**: tell the fly
what things are like, and it learns while you talk. The fly also lives on a little loop of
three places and can fly around it.

There are **20 Zumi words**, the grammar is two rules, and any other word is a *thing* you
can talk about. You can learn it in fifteen minutes.

Try it:

```bash
python -m chatfly.zumi configs/phase6_zumi.yaml            # fast rate model
python -m chatfly.zumi configs/phase6_zumi.yaml --sim spiking
python -m chatfly.web configs/phase6_zumi.yaml               # in the browser: http://127.0.0.1:8000
```

---

## 1. Sounds

Every Zumi word is one short syllable. Vowels sound as in Spanish or Japanese: *a* as in
"father", *e* as in "bed", *i* as in "ski", *o* as in "go", *u* as in "rule". `bzz` is a buzz.
Capital letters and punctuation do not matter: `Fire ka mal.` = `fire ka mal`.

## 2. Part one: talking about anything

### Things

Any word that is not a Zumi word is a **thing**: `fire`, `sugar`, `cat`, `moon`, `pizza`,
`grandma`. It must be one word made of letters only (`icecream`, not `ice cream` or `r2d2`).
The fly has never heard of anything when the conversation starts.

To the fly, each new word is like a new smell: a fixed pattern of activity on its smell
input neurons. That is what lets it learn about things it has never met.

### Properties

The fly can learn four pairs of opposites:

| word | meaning | | word | meaning |
|---|---|---|---|---|
| `bon` | good | ↔ | `mal` | bad |
| `gu` | big | ↔ | `pi` | small |
| `ho` | hot | ↔ | `ku` | cold |
| `ra` | fast | ↔ | `lo` | slow |

Memory hooks: **bon**bon is good · **mal**ware is bad · **gu**lliver is big · **pi**xie is
small · **ho**t · **ku**ld (cold) · **ra**ce · **lo**w and slow.

### Three sentences

| you say | means | the fly |
|---|---|---|
| `fire ka mal ho.` | fire **is** bad and hot | learns it, says `o` |
| `fire ho ba?` | is fire hot? | `ya`, `na`, or `hm` |
| `fire ke?` | what is fire (like)? | lists what it knows: `mal ho`, or `hm` |

- `ka` means "is". After it come one or more properties: `sun ka gu ho bon.`
- A thing can't be both sides of one pair: `fire ka ho ku.` gets `e`.
- You can change your mind. `fire ka bon.` later makes the fly relearn: fire is now good.
  It keeps the other things it knew about fire.
- `ke` asks about all four pairs and the fly names only the ones it is sure about.

### Example

```
> fire ke?            # what is fire?
fly: hm               # I don't know anything about fire yet
> fire ka mal ho.     # fire is bad and hot
fly: o
> ice ka ku.          # ice is cold
fly: o
> fire ke?
fly: mal ho           # bad, hot
> fire ku ba?         # is fire cold?
fly: na
> ice ho ba?          # is ice hot?
fly: na
> ice bon ba?         # is ice good?
fly: hm               # you never said
```

### How much can it remember?

It depends on how many things you teach it. Each thing's smell lights up a small set of
Kenyon cells, and the more things there are, the more those sets overlap. Past some point,
the fly starts answering questions it was never told about. On test flies it never saw
during tuning, with 20% of facts changed partway through:

| model | things taught | answers right | taught facts it answers | never-taught questions it answers anyway |
|---|---|---|---|---|
| rate | 40 | 100% | 100% | 0.2% |
| rate | 80 | 100% | 96% | 2.1% |
| rate | 160 | 99.7% (worst fly 98.8%) | 82% | 11% |
| spiking | 40 | 100% | 80% | 0.4% |
| spiking | 80 | 100% | 76% | 5.8% |

So a conversation about **a few dozen things** works well. The spiking fly says `hm` more
often: about a fifth of the facts it was taught come back as "not sure" rather than wrong.

## 3. Part two: the fly's trip

### The fly's world

```
            ne (nest)
           ↗         ↘
    po (pond)  ←——  fi (flower)

   zu = fly ON along the arrows      ki = fly BACK against them
```

- Going **on** (`zu`): nest → flower → pond → nest → …
- Going **back** (`ki`): nest → pond → flower → nest → …
- Every conversation starts at the **nest**.

The fly remembers where it is from one sentence to the next. That memory lives in the fly
itself: its answer to the last step is fed back into it as the next step's starting place.

### Trip words

| word | kind | meaning | memory hook |
|---|---|---|---|
| `ne` | place | the nest | **ne**st |
| `fi` | place | the flower | **fi**eld of flowers |
| `po` | place | the pond | **po**nd |
| `zu` | move | fly on (one place forward) | *zoom* |
| `ki` | move | fly back (one place backward) | *kick back* |
| `ta` | move | stay where you are | *ta*rry |
| `nu` | question | where are you? | *now*? |
| `si` | mood | "what if…" (imagine it; do not really fly) | *si* = "if" in Spanish |
| `bzz` | restart | start over: back to the nest | a buzz hello |

`ba` works here too: `fi ba?` = "are you at the flower?"

### Trip grammar

Every trip sentence is these parts, **in this order**, each one optional:

```
[ bzz ]  [ si ]  [ moves … ]  [ question ]
 restart  what-if  zu ki ta    nu   or   <place> ba
```

- **Moves come before the question.** `zu nu?` is right; `nu zu?` gets `e`.
- **The question comes last**: `nu` alone, or a place followed by `ba`.
- **`si` needs a question.** `si zu nu?` is fine; plain `si zu.` gets `e`.
- **Chain as many moves as you like**: `zu zu ki ta zu nu?`

| you say | type | what the fly does |
|---|---|---|
| `zu ki.` | telling | flies, answers `o` |
| `zu nu?` | asking where | flies, then names its place |
| `zu fi ba?` | asking yes/no | flies, then `ya` or `na` |
| `si zu nu?` | wondering | imagines flying, names the place, **stays put** |

### Trip examples

```
> nu?            # where are you?
fly: ne          # nest
> zu.            # fly on
fly: o
> zu po ba?      # fly on — are you at the pond now?
fly: ya
> si ki nu?      # what if you flew back — where would you be?
fly: fi          # at the flower (but it did not move)
> nu?
fly: po          # still at the pond
> bzz zu zu zu nu?    # start over, once around the loop
fly: ne
```

If the fly's answer for any step was too weak to trust, it says `hm` and stops pretending;
everything after is `hm` until you say `bzz`.

## 4. The fly's words

| fly says | meaning |
|---|---|
| `ne` / `fi` / `po` | I am (or would be) at the nest / flower / pond |
| `bon`, `mal ho`, … | the properties it remembers |
| `ya` | yes |
| `na` | no |
| `o` | okay, done (or learned) |
| `hm` | I'm not sure: never told, or I lost track |
| `e` | huh? I don't understand that sentence |

## 5. Exercises

Start from a fresh conversation. Answers are below.

1. Teach the fly that the sun is big, hot and good.
2. Ask whether the sun is cold.
3. Ask what a snail is like, then teach it that snails are small and slow, then ask again.
4. `zu zu nu?`
5. `si ki ki ne ba?`, then `nu?`
6. What does the fly say to `fire ka ho ku.`? Why?
7. What does the fly say to `zu ka bon.`? Why?

<details><summary>Answers</summary>

1. `sun ka gu ho bon.` (fly: `o`)
2. `sun ku ba?` (fly: `na`)
3. `snail ke?` (fly: `hm`), `snail ka pi lo.` (fly: `o`), `snail ke?` (fly: `pi lo`)
4. `po` (nest → flower → pond)
5. `na` (back twice from the nest = flower), then `ne` (it only imagined it)
6. `e`: hot and cold are one pair, so a thing can't be both.
7. `e`: `zu` is a Zumi word, not a thing.

</details>

## 6. Cheat sheet

```
things   any other word: fire, sugar, cat, moon …
props    bon good / mal bad · gu big / pi small · ho hot / ku cold · ra fast / lo slow
teach    thing ka prop…        fire ka mal ho.
ask      thing prop ba?        fire ho ba?       thing ke?   fire ke?

places   ne nest · fi flower · po pond          ring: ne →zu→ fi →zu→ po →zu→ ne
moves    zu on · ki back · ta stay
ask      nu = where?   <place> ba = there?    si … = what if      bzz = start over
order    [bzz] [si] moves… [nu | place ba]

fly      ya yes · na no · o ok · hm unsure · e huh?
```

## 7. What is the fly, and what is the interface?

To keep the fly honest (`CLAUDE.md` principle 2), the parts are split like this:

| who | does |
|---|---|
| **the interface** (`src/chatfly/zumi.py`) | splits your sentence into words by the grammar; turns each thing into its fixed input pattern; for `ka`, gives the fly dopamine in the right output pair; for questions, reads the fly's output and names it; feeds move words through the fly's loop; keeps or drops the result for `si`; resets on `bzz`. It **stores no facts** and never looks at the map of the ring. |
| **the fly** (simulated mushroom body) | *Things:* learns during the conversation, by dopamine-gated plasticity at its KC→MBON synapses, which side of each pair a smell goes with, and answers from those synapses. *Trip:* learned beforehand what each move does from each place, and computes every step itself. |

Each property pair is its own set of KC→MBON synapses with two outputs, like separate
compartments of the mushroom body, all reading the same KCs. A thing's smell activates a
small, sparse set of KCs, and teaching weakens those KCs' drive to the opposite output.

Controls (`docs/notes/phase6-zumi.md`): a fly whose teacher is shuffled answers at chance,
and the same learning rule without the KC layer remembers far less and answers questions it
was never taught.

### Why the trip world is so small

The fly learns a move well only if its right answer isn't outvoted by the answers of the
other moves from the same place and the same move from other places (Phase 4 failure
mechanism). Moves that shuffle the places help; a "go home" word breaks it (4 of 10 seeds).
Four places were learned on 8 of 10 seeds, five on 5 of 10. Three places and three moves
are what the fly learns every time, so `bzz` is an interface rule, not a learned word.
