# Implementation Plan: Report v2

## Overview

Rewrite `report.ipynb` and supporting code to match the spec in `docs/report_spec.md`. The current report is a variant gallery; the new report has a toy model section, a full model section with interventions and sensitivity analysis, and appendices.

## Step 1: Full-model payoff function

**File**: `src/report/primitives.py`

Implement the full payoff with $\Omega$, $\tilde{A}$, and $z$:

```
payoff_i = E[ Σ_j Ω_j · Ã_ij ]
```

where the expectation is over all $2^n$ alignment outcome vectors, weighted by copula probabilities.

For each outcome vector $\mathbf{a} \in \{0,1\}^n$ (1 = aligned):
- $\Omega_j^{aligned} = (R_j c_j)^w$, $\Omega_j^{misaligned} = (R_j c_j)^{w(1-z)}$
- Normalize: $\Omega_j = \text{raw}_j / \sum_k \text{raw}_k$
- $\tilde{A}_{ij} = A_{ij}$ if aligned, 0 if misaligned
- Weight by P(outcome) from copula

Computing all $2^n$ outcomes is feasible for n=5 (32 outcomes). For the copula probability of each outcome, use inclusion-exclusion or direct computation from the Gaussian copula.

**Key function**: `full_model_payoff(s_all, R, A, k_values, alpha, w, z, delta, rho) -> list[float]`

**Also need**: `full_model_nash(R, A, k_values, alpha, w, z, delta, rho) -> list[float]` using iterated best response.

**Also need**: `expected_human_share(s_all, R, A, k_values, alpha, w, z, delta, rho) -> float` — the expected value of $\sum_j \Omega_j \cdot \mathbb{1}[j \text{ aligned}]$, which is the fraction of the universe controlled by aligned AI (independent of any lab's preferences).

## Step 2: Default parameters module

**File**: `src/report/defaults.py`

Centralize all default parameter values from `docs/parameters.md` and `docs/report_spec.md`:
- R vector, A matrix, lab names
- w=2.0, k/epsilon calibrated, delta=0.5, r=0.5, z=1
- Parameter ranges for sensitivity analysis

## Step 3: Toy model section

**File**: `src/report/toy_model.py` (data), update `src/report/plots.py`

Reuse existing code but simplify:
- Plot s* and P² vs k for n=2 symmetric, α=0.466 only (not comparing α=1)
- Markdown explaining the closed form from `docs/closed_forms.md`: $p \approx 1 - \sqrt{w(n-1)/(nk)}$, joint survival $\approx 1 - \sqrt{nw(n-1)/k}$
- Brief note on general w closed form and how winner-take-all intensifies racing

## Step 4: Full model — default outcomes plot

**File**: `src/report/full_model.py` (data), update `src/report/plots.py`

- Compute Nash equilibrium for the 5-lab model with default R, A, w, z, δ, r
- Plot expected human share vs k (x-axis), with default ε=0.466
- Show each lab's equilibrium safety fraction
- Save to `plots/full_model_default.png`

## Step 5: Intervention bar graph

**File**: `src/report/interventions.py`

Compute Nash equilibrium and expected human share for each intervention, relative to baseline:

1. **Remove China**: n=4, remove China from R (renormalize) and A
2. **Duplicate every lab**: n=10, each lab appears twice with half resources
3. **Secretly make largest lab 100% safe**: Fix GDM's s=1 (others don't know, so they play as if GDM is optimizing normally). This is a Stackelberg-like setup — GDM commits to s=1 but others best-respond to their beliefs about GDM.
4. **Make largest lab 100% safe, tell only them**: GDM knows s_GDM=1, others don't know. GDM optimizes knowing they're safe; others play Nash assuming GDM optimizes normally.
5. **Double largest lab's resources**: Double GDM's R, renormalize
6. **Increase amity 10% toward 1**: A' = A + 0.1*(J - A) where J is all-ones
7. **Make safety a public good (δ=1)**: Set delta=1

Plot: horizontal bar graph, baseline (dotted vertical line), bars showing change from baseline. Save to `plots/interventions.png`.

**Note on "secretly" interventions**: These require computing equilibrium where some players have different information. For "secretly safe," fix s_GDM=1 and have others best-respond as if GDM is a normal player (they don't observe s_GDM=1). For "tell only them," GDM knows their alignment is guaranteed and optimizes accordingly (spending 0 on safety, all on capabilities), but others don't know this.

## Step 6: Sensitivity analysis

**File**: `src/report/sensitivity.py`

Follow the takeoff model format:
1. Define triangular prior distributions for each parameter:
   - w: range [1.0, 5.0], mode 2.0
   - δ: range [0, 0.75], mode 0.5
   - r: range [0, 1], mode 0.5
   - z: range [0.5, 1], mode 1.0
2. Sample 200-500 parameter vectors (all params drawn independently)
3. For each sample, compute Nash equilibrium and expected human share
4. For each parameter, bucket into 10 deciles, compute conditional median expected human share
5. Plot: 2x2 grid of subplots, one per parameter. Steelblue line-and-dot showing conditional median vs parameter value.
6. Save to `plots/sensitivity.png`

**Performance concern**: Each Nash solve with r>0 takes ~23ms (after copula optimization). 500 samples × ~23ms ≈ 12s. Feasible but should parallelize if needed.

## Step 7: Notebook assembly

**File**: `report.ipynb`

Structure:
1. **Toy model** — markdown + plots from Step 3
2. **Full model** — markdown explaining the model, default outcomes (Step 4), interventions (Step 5), sensitivity (Step 6)
3. **Appendix A**: Parameter choices — pull from `docs/parameters.md`
4. **Appendix B**: When is computing Nash equilibrium easy? — pull from `docs/closed_forms.md`
5. **Appendix C**: Why Gaussian copula? — write qualitative discussion (no experiments needed per spec)

## Step 8: Verification

- `ruff format . && ruff check .`
- `pytest tests/`
- Run notebook end-to-end, verify plots saved to `plots/`

## Execution order

Steps 1-2 are foundational (everything depends on them). Steps 3-6 are mostly independent and can be parallelized. Step 7 depends on all prior steps. Step 8 is final.

```
1 (payoff fn) ──┐
2 (defaults)  ──┼──> 3 (toy model) ──┐
                ├──> 4 (full model) ──┤
                ├──> 5 (interventions)┼──> 7 (notebook) ──> 8 (verify)
                └──> 6 (sensitivity) ─┘
```

## Open questions

1. **2^n copula computation**: For n=5 we need P(outcome vector) for all 32 outcomes. The Gaussian copula gives the joint CDF P(all ≤ thresholds), but we need P(specific subset aligned, rest misaligned). This requires inclusion-exclusion over the copula CDF. For n=5 this is 32 terms, each requiring a copula evaluation — feasible but needs careful implementation.

2. **"Secretly safe" equilibrium**: How exactly to model information asymmetry? Simplest: fix s_GDM and have others play Nash as if GDM is optimizing (they use the original payoff structure). GDM's actual safety is revealed only when computing the outcome.

3. **Expected human share vs survival probability**: The spec says to explain that survival probability can be interpreted as expected human share. With the full Ω model and z<1, these are different quantities. Need to be clear about which we're plotting.
