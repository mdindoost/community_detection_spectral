# DSpar source paper — definitive notes (settles C1; no more guessing)

Source: `PAPER_RESTRUCTURE/references/dspar_source_paper.pdf`
Liu, Zhou, Jiang, Li, Chen, Choi, Hu. "DSpar: An Embarrassingly Simple Strategy for Efficient GNN
Training and Inference via Degree-Based Sparsification." **TMLR, 07/2023** (our refs.bib entry is
already correct). Code: github.com/warai-0toko/DSpar_tmlr.

## The exact theory (what we may cite, verbatim-accurate)

**Setting (p.4, Eq. 3).** Their effective resistance is defined via the NORMALIZED Laplacian
$\mathcal{L} = I - D^{-1/2} A D^{-1/2}$:  $R_e = (\mathcal{X}_u - \mathcal{X}_v)^\top \mathcal{L}^+ (\mathcal{X}_u - \mathcal{X}_v)$.
$\alpha$ ($\le 2$) denotes the smallest non-zero eigenvalue of $\mathcal{L}$ (the normalized
spectral gap). Note: NOT the combinatorial Laplacian our Background defines.

**Their Theorem 1 (p.4; = Lovász 1993, Cor. 3.3).**  For all $e=(u,v)$:
$\tfrac{1}{2}(\tfrac{1}{d_u} + \tfrac{1}{d_v}) \le R_e \le \tfrac{1}{\alpha}(\tfrac{1}{d_u} + \tfrac{1}{d_v})$.
Degree scores sandwich effective resistance within a factor $2/\alpha$. "The bound is tight for
well-connected graphs." Their own intuition (p.4): in graphs with cluster structure the random
walk is trapped in clusters — i.e., $\alpha$ is small exactly when community structure is strong.

**Their Theorem 2 (p.4–5; proof App. B, p.15–17).**  With $Q = O(|V| \log |V| / \epsilon^2)$
samples drawn WITH replacement with $p_e \propto \tfrac{1}{d_u} + \tfrac{1}{d_v}$, and retained
edges REWEIGHTED $w_e = A_e/(Q p_e)$ (their Algorithm 1, p.3; $\mathbb{E}[A'] = A$, unbiased),
the sparsified graph satisfies, for all $x$:
$(1 - \tfrac{\epsilon}{\alpha})\, x^\top \mathcal{L} x \;\le\; x^\top \mathcal{L}' x \;\le\; (1 + \tfrac{\epsilon}{\alpha})\, x^\top \mathcal{L} x$,
hence eigenvalues preserved to $(1 \pm \epsilon/\alpha)$ (their Eq. 5). Proof is the
Spielman–Srivastava machinery with $p'_e \ge (\alpha/2) p_e$ (their Eq. 15).

**THE KEY NUANCE: the effective error is $\epsilon/\alpha$.** The guarantee weakens as the
normalized spectral gap shrinks — that is, PRECISELY on graphs with strong community structure.
They acknowledge this operationally (p.9, obs. 3): "$Q$ should scale with the graph connectivity
... in practice, we absorb the term $\lambda/\alpha$ into $\epsilon$."

**Their Theorem 3 (p.5).**  GNN embedding perturbation: $\|H^{(l+1)} - H'^{(l+1)}\|_F < \epsilon \tfrac{\lambda_1}{\alpha}\|H^{(l)}\Theta^{(l)}\|_F$.

## Their empirical claims (p.8–11)

- Accuracy: "negligible drop ($\approx 0.3\%$)" across 5 datasets/4 models (Table 1); sometimes
  "even better accuracy," attributed to a REGULARIZATION effect citing DropEdge (p.9, obs. 2).
  This is a published improvement claim for GNN accuracy — usable as an antecedent in our intro.
- Sparsification rates 25–95% (Table 2); sampling overhead seconds (Table 3).
- **Spectral preservation experiment (§5.2.2, p.10, Fig. 5)**: cluster-relevant eigenvalues of
  $\mathcal{L}$ preserved to $\lesssim 1.5\%$ relative error under DSpar (vs. up to ~100% for
  random sparsification at the other end of the spectrum); their words: "the cluster structure is
  well-preserved by DSpar." NOTE: this is measured on the WEIGHTED sparsified graph and their
  top/bottom eigenvalue terminology is used loosely — cite the claim, not their labeling.

## What this settles for OUR paper

1. **C1 wording (approved facts):** "DSpar proves that degree scores sandwich (normalized)
   effective resistance within a factor $2/\alpha$ [their Thm 1, after Lovász], so with-replacement
   sampling with importance reweighting yields a $(1 \pm \epsilon/\alpha)$ spectral approximation
   of the normalized Laplacian [their Thm 2]; the guarantee is attached to the weighted graph and
   weakens as the normalized spectral gap $\alpha$ shrinks."
2. **The story's sharpest new point:** the DSpar–spectral equivalence is strongest on expanders
   and DEGRADES exactly when community structure is strong (small $\alpha$). So even inside the
   weighted regime, the preservation guarantee is weakest on the graphs communities live in. This
   deepens the regime-split narrative rather than threatening it.
3. **Their own experiment supports C2's spirit:** cluster-relevant spectra preserved (weighted).
   Our C2 proposition (fixed-partition modularity preserved to $O(\epsilon/\alpha)$ — note the
   $\alpha$!) should be stated with THEIR error factor, and can cite §5.2.2 as consistent evidence.
4. **They never claim community-detection improvement.** Preservation ($\pm 0.3\%$ accuracy) plus
   an accuracy-improvement note attributed to regularization. Our abstract's "stronger claim ...
   advanced for degree-based sparsifiers" is supportable via their better-accuracy observation,
   but must be phrased as GNN accuracy, not community detection.

## Corrections needed in OUR tex (NOT yet applied — story-first discipline)

| # | Location | Current (wrong/imprecise) | Correct per source |
|---|---|---|---|
| T1 | §4 sampler audit | "The originally published DSpar procedure draws $q = \lceil \alpha m \rceil$ edge samples" | Liu et al. draw $Q = |V|\log|V|/\epsilon^2$ samples ($\epsilon$-controlled); the $q = \lceil \alpha m \rceil$ convention is the community-detection adaptation audited here (our repo's), not theirs. Attribute correctly. |
| T2 | §2 DSpar sentence | "under conditions relating degrees to effective resistances ... approximates effective-resistance sparsification" | State precisely: factor-$2/\alpha$ sandwich (their Thm 1) and $(1\pm\epsilon/\alpha)$ guarantee on the weighted graph (their Thm 2); normalized Laplacian. |
| T3 | §6.1 reconciliation | "(1±ε) sparsifier preserves ... to O(ε)" | Use $(1 \pm \epsilon/\alpha)$ and note the guarantee weakens for small-$\alpha$ (community-rich) graphs — strengthens our point. |
| T4 | §2 Background | Our $R_{\mathrm{eff}}$ discussion is combinatorial-Laplacian; their theory is normalized | One clause: "for the normalized Laplacian" where we invoke their result. |
| T5 | Intro antecedent | "reports preserved downstream accuracy at scale" | Accurate; may add "with occasional accuracy improvements attributed to regularization" + their §5.2.2 spectral-preservation experiment as the strongest antecedent for "preservation." |
| T6 | C2 proposition (to be written) | O(ε) | $O(\epsilon/\alpha)$, inheriting their error factor. |

## For the STORY (updates to STORY.md arc)

- Arc point 1 gains precision: the question "spectra are preserved — are communities?" now has a
  built-in tension from the source itself: the preservation guarantee is weakest exactly where
  communities are strongest ($\epsilon/\alpha$). The question was even better than we knew.
- Arc point 3 (regime split) gains a third shade: weighted-and-well-connected (strong guarantee) /
  weighted-but-clustered (weak guarantee, empirically still good per their §5.2.2) /
  unweighted (no guarantee; our artifacts live here).
