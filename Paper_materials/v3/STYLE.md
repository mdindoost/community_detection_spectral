# v3 style rules — HARD CONSTRAINTS

Set by Mohammad, 2026-07-26. These are not preferences. Every sentence of the paper must comply.

## 1. No em-dashes. Anywhere. Ever.

No `---`, no `--` used as a dash, no unicode em-dash or en-dash used as punctuation.
Use a comma, a semicolon, a colon, parentheses, or two sentences. If a sentence seems to need an
em-dash, it is usually a sentence that should be split.

(En-dashes remain correct inside numeric ranges written in math mode or in citation ranges,
e.g. `10--50` for a numeric range in text is acceptable ONLY where it is a range of numbers, never
as punctuation between clauses. When in doubt, write "10 to 50".)

## 2. No AI-characteristic phrasing

Banned constructions, because they read as machine-written:

- "It is not just X, it is Y." / "This is not X. It is Y." (the negation-reversal flourish)
- "Importantly," / "Notably," / "Crucially," / "It is worth noting that"
- "delve", "underscore", "showcase", "highlight" (as a verb for "show"), "pivotal", "robust"
  as a filler adjective, "leverage" as a verb, "landscape" as a metaphor
- Three-item lists used for rhythm rather than because there are exactly three things
- Paragraphs that open with a thesis sentence, give three parallel examples, and close by
  restating the thesis
- "In this section, we will..." roadmapping at the start of every section
- Rhetorical questions used as transitions
- Sentence fragments for emphasis
- Editorial self-praise about the work ("the most informative result", "strikingly", "remarkably")

## 3. Positive style targets

- Declarative sentences. Subject, verb, object.
- Numbers do the emphasis; adjectives do not.
- Vary sentence length naturally. Do not alternate short/long mechanically.
- Say what was measured, then what it means. Not the reverse.
- Where a claim is limited, state the limit in the same sentence, not in a later hedge.

## 4. Enforcement

Before any section is committed, grep it:

```
grep -n -- "---" sections/*.tex          # must return nothing
grep -nE "Importantly|Notably|Crucially|delve|underscore|showcase|pivotal|it is worth noting" sections/*.tex
```

Both must come back empty.
