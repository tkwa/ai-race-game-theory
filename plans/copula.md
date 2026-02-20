# Copula comparison subreport

## Goal

Determine whether copula choice meaningfully affects model conclusions, or mostly just shifts joint survival up/down uniformly.

## Copulas to compare

1. **Gaussian** (current) — no tail dependence, symmetric
2. **Student-t** (df=4) — symmetric tail dependence (correlated extremes)
3. **Clayton** — lower tail dependence (correlated failures more likely than correlated successes)
4. **Gumbel** — upper tail dependence (correlated successes more likely)
5. **Frank** — symmetric, no tail dependence (similar shape to Gaussian but different)

For each, map equicorrelation r to the copula's native parameter via Kendall's tau: tau = (2/pi)*arcsin(r), then invert the copula-specific tau-theta relationship.

## Experiments

### A. Baseline comparison
n=5, r=0.5, uniform p=0.9. Table of joint survival per copula.

### B. Sensitivity to r
Sweep r from 0 to 0.95, n=5, p=0.9. Plot joint survival vs r for each copula. Does the ordering change?

### C. Sensitivity to p
Sweep p from 0.5 to 0.99, n=5, r=0.5. Plot joint survival vs p. Do copulas diverge more at low or high p?

### D. Asymmetric marginals
n=5, r=0.5, probs=[0.99, 0.95, 0.9, 0.8, 0.7]. Table per copula.

### E. Marginal value of safety investment (genuinely uncertain)
Compute d(joint_survival)/dp_1 at p_1=0.9 with others at p=0.8, for each copula and r in {0.2, 0.5, 0.8}. This tests whether copula choice changes the *strategic incentive* to invest in safety — i.e., does Clayton (correlated failures) make individual effort less valuable because "if others fail, I fail too"? Or more valuable because pushing above the failure cluster matters more?

## Implementation

- Code in `src/copula/`
- Use Monte Carlo (100k samples) for Clayton, Gumbel, Frank; scipy multivariate_t for t-copula
- Plots saved to `docs/copula_plots/`
- Writeup in `docs/copula.md`

## Output

A short writeup answering: does copula choice mostly act as a uniform shift in survival probability, or does it change which parameters matter most? If the former, Gaussian is fine and we just note the sensitivity. If the latter, document which conclusions are copula-dependent.
