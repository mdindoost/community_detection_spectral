# v3 — file map and the Mohammad/Claude sync protocol

**This folder is self-contained.** Upload `Paper_materials/v3/` to Overleaf as-is and it compiles;
nothing is referenced outside it.

---

## File map

```
Paper_materials/v3/
├── main.tex                      <- preamble, title, section order, \input calls
├── refs.bib                      <- bibliography (47 entries). SELF-CONTAINED COPY.
│                                    NOTE: Paper_materials/refs.bib (parent) is v1/v2's copy.
│                                    From now on, v3/refs.bib is the one that matters.
├── sections/
│   └── 01_introduction.tex       <- DRAFTED
│       (02_background, 03_protocol, 04_below, 05_boundary,
│        06_mechanism, 07_related, 08_discussion to come; abstract
│        and conclusion are written LAST, inside main.tex)
├── figures/                      <- empty; figures land here as sections need them
├── tables/                       <- empty; generated tables land here
└── main.pdf                      <- build output (do not edit)
```

**Compile locally:** `cd Paper_materials/v3 && ~/.local/bin/tectonic -X compile main.tex`

---

## Sync protocol (we are both editing)

Mohammad works in Overleaf; Claude works in the local git repo. The risk is both of us editing the
same file and one overwriting the other. Rules:

1. **One owner per file at a time.** Before Claude edits a section, he says which file. Before
   Mohammad edits one in Overleaf, he says which file. Whoever is not the owner does not touch it.
2. **Section files are the unit**, which is why the prose lives in `sections/*.tex` rather than in
   `main.tex`. Two people can work on different sections with no conflict.
3. **`main.tex` is low-traffic** — it changes only when a section is added or the order changes.
   Claude edits it; Mohammad should avoid it unless he says otherwise.
4. **`refs.bib` is append-only for both of us.** New entries go at the end with a comment naming
   who added them and why. Never reorder or reformat the file; that is what creates conflicts.
5. **After every Claude edit**, the changed `.tex`/`.bib` are committed and pushed to
   `refactor_v2` immediately, so Mohammad can pull or copy the current text into Overleaf.
6. **When Mohammad edits in Overleaf**, he pastes the changed section back (or says "I changed
   §N") so the repo stays the source of truth. If the repo and Overleaf ever disagree, **Overleaf
   wins for prose Mohammad wrote, the repo wins for numbers**, and we reconcile explicitly rather
   than silently.

---

## Status

| Section | State | Owner |
|---|---|---|
| Abstract | write LAST | — |
| 1 Introduction | **drafted, compiles** (~1600 words) | open for Mohammad's edits |
| 2 Background and setting | not started | — |
| 3 Honest evaluation (protocol) | not started | — |
| 4 Below the boundary | not started | — |
| 5 The boundary | not started | — |
| 6 Mechanism | not started | — |
| 7 Related work | not started | — |
| 8 Discussion | not started | — |
| 9 Conclusion | write LAST | — |

Framing, weights and the one-sentence claim: `PAPER_RESTRUCTURE/phase4_v3/PAPER_STRUCTURE.md`.
Introduction beat plan and settled decisions: `PAPER_RESTRUCTURE/phase4_v3/INTRO_PLAN.md`.
Fixed terminology (do not drift): `PAPER_RESTRUCTURE/TERMINOLOGY.md`.
