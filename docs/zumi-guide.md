# Zumi — a guidebook

Zumi is a tiny language for talking with the digital fly (a connectome-style computer model
of the fly mushroom body, not a living fly). It has **10 words you say** and **8 words the fly
says**. You can learn it in ten minutes.

Try it:

```bash
python -m chatfly.zumi configs/phase6_zumi.yaml            # fast rate model
python -m chatfly.zumi configs/phase6_zumi.yaml --sim spiking
```

---

## 1. The fly's world

The fly lives on a little loop of three places:

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

## 2. Sounds

Every word is one syllable. Letters sound as in Spanish or Japanese: *a* as in "father",
*e* as in "bed", *i* as in "ski", *o* as in "go", *u* as in "rule". `bzz` is a buzz.
Capital letters and punctuation do not matter: `Zu, zu. Fi ba?` = `zu zu fi ba`.

## 3. Your words

| word | kind | meaning | memory hook |
|---|---|---|---|
| `ne` | place | the nest | **ne**st |
| `fi` | place | the flower | **fl**ower / **fi**eld |
| `po` | place | the pond | **po**nd |
| `zu` | move | fly on (one place forward) | *zoom* |
| `ki` | move | fly back (one place backward) | *kick back* |
| `ta` | move | stay where you are | *ta*rry |
| `nu` | question | where are you? | *now*? |
| `ba` | question | "… right?" — turns a place into a yes/no question | |
| `si` | mood | "what if…" (imagine it; do not really fly) | *si* = "if" in Spanish |
| `bzz` | greeting | start over: back to the nest | a buzz hello |

## 4. The fly's words

| fly says | meaning |
|---|---|
| `ne` / `fi` / `po` | "(I am / I would be) at the nest / flower / pond" |
| `ya` | yes |
| `na` | no |
| `o` | okay, done |
| `hm` | I'm not sure — I lost track of where I am |
| `e` | huh? — I don't understand that sentence |

## 5. Grammar: one rule

Every sentence is these parts, **in this order**, each one optional:

```
[ bzz ]  [ si ]  [ moves … ]  [ question ]
 restart  what-if  zu ki ta    nu   or   <place> ba
```

That is all. Some consequences:

- **Moves come before the question.** `zu nu?` is right; `nu zu?` gets `e`.
- **A question is the last thing.** Either `nu` alone or a place followed by `ba`.
- **`si` needs a question.** "What if I fly on?" only makes sense if you ask something
  about it: `si zu nu?`. Plain `si zu.` gets `e`.
- **You can chain as many moves as you like**: `zu zu ki ta zu nu?`

### The four kinds of sentence

| you say | type | what the fly does |
|---|---|---|
| `zu ki.` | telling | flies, answers `o` |
| `zu nu?` | asking where | flies, then names its place |
| `zu fi ba?` | asking yes/no | flies, then `ya` or `na` |
| `si zu nu?` | wondering | imagines flying, names the place, **stays put** |

## 6. Example conversations

Comments after `#` are English glosses.

**A first flight**

```
> nu?            # where are you?
fly: ne          # nest
> zu.            # fly on
fly: o
> nu?
fly: fi          # flower
> zu po ba?      # fly on — are you at the pond now?
fly: ya
```

**Imagining**

```
> nu?
fly: po
> si ki nu?      # what if you flew back — where would you be?
fly: fi          # at the flower (but it did not move)
> nu?
fly: po          # still at the pond
```

**Long trips**

```
> bzz zu zu zu nu?    # start over, fly on three times: around the whole loop
fly: ne
> ki ki ne ba?
fly: na               # back twice from the nest = flower
```

**When the fly is unsure**

If the fly's internal answer for any step was too weak to trust, it says `hm` and stops
pretending. Everything after that is `hm` too, because it no longer knows where it is.
Say `bzz` to start over. (Illustration; whether and where this happens depends on the fly.)

```
> zu ki zu ki zu nu?
fly: hm
> nu?
fly: hm
> bzz nu?
fly: ne
```

## 7. Exercises

Start each one from the nest (`bzz`). Answers are below.

1. `zu zu nu?`
2. `ki nu?`
3. `zu ki ta zu fi ba?`
4. `si ki ki ne ba?`, then `nu?`
5. How do you get from the nest to the pond in one move?
6. What does the fly say to `fi nu?` Why?
7. Say "What if you flew on twice — would you be at the pond?"

<details><summary>Answers</summary>

1. `po` (nest → flower → pond)
2. `po` (back from the nest is the pond)
3. `ya` (on, back, stay, on = one step on = flower)
4. `na` (back twice = flower), then `ne` (it only imagined it)
5. `ki`
6. `e`: a place word is only allowed right before `ba`.
7. `si zu zu po ba?` (answer: `ya`)

</details>

## 8. Cheat sheet

```
places   ne nest · fi flower · po pond        ring: ne →zu→ fi →zu→ po →zu→ ne
moves    zu on · ki back · ta stay
ask      nu = where?   <place> ba = there?    si … = what if
other    bzz = start over
fly      ya yes · na no · o ok · hm unsure · e huh?
order    [bzz] [si] moves… [nu | place ba]
```

## 9. What is the fly, and what is the interface?

To keep the fly honest (`CLAUDE.md` principle 2), the parts are split like this:

| who | does |
|---|---|
| **the interface** (`src/chatfly/zumi.py`) | splits your sentence into words by the one grammar rule; feeds each move word to the fly; turns the fly's final place into a word; compares that place with the one in a `ba` question; keeps or drops the result for `si`; resets on `bzz` |
| **the fly** (simulated mushroom body) | learned, by dopamine-gated plasticity, what each move does from each place, and computes every step itself: (current place, move) → next place |

The interface never looks at the map of the ring; only the trainer and the evaluation do.
A control fly trained with a shuffled teacher, behind the same interface, answers at chance
(about 1 in 3; `docs/notes/phase6-zumi.md`).

### Why the world is so small, and why there is no "go home" word

The fly learns well only if, for each place and each move, the right answer is not
outvoted by the answers of the other moves from the same place and the same move from other
places (Phase 4 failure mechanism, `docs/notes/phase4-gate-d.md`). Moves that **shuffle the
places** (every move from every place leads somewhere different) satisfy this. A word like
"go home" breaks it: tested, the fly learned it on only 4 of 10 seeds. A loop of four places
was learned on 8 of 10 seeds in the rate model, five places on 5 of 10. Three places and
three moves are what the fly learns reliably. That is why `bzz` (start over) is an interface
rule, not a word the fly learned.
