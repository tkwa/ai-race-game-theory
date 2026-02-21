# When Does Merging Help?

## The equity-share model

The correct way to model merging is through equity shares. When labs in set $M$ merge:

1. The merged entity has resources $R_M = \sum_{i \in M} R_i$ and makes a single safety choice $s_M$.
2. Each original lab $i \in M$ receives equity share $e_i$ (typically $e_i = R_i / R_M$).
3. If the merged entity's ASI is aligned, its controlled resources $\Omega_M$ are split by equity: lab $i$ gets share $e_i \cdot \Omega_M$, valued at the original amity $A_{ij}$ toward each co-owner $j$.
4. Lab $i$'s effective amity toward the merged entity is $\sum_{j \in M} e_j A_{ij}$ — the equity-weighted sum of its amity toward each co-owner's portion.

This means the amity matrix for the merged game has:
- $\tilde{A}_{i,M} = \sum_{j \in M} e_j A_{ij}$ for any lab $i$ (including outside labs).
- Lab $i$'s payoff uses its own amity values, not the merged entity's averaged values.

## Pure Tullock contest (no alignment)

In a pure Tullock contest with proportional equity ($e_i = R_i / R_M$):

**Theorem.** With $w > 1$, if the merging parties have equal resources and there is at least one outside competitor, merging is individually profitable for every member.

*Proof sketch.* For $m$ equal-sized labs with resource $R$ merging against outside competition $Z = \sum_{j \notin M} R_j^w > 0$:

$$\frac{e_i \cdot (mR)^w}{(mR)^w + Z} = \frac{R^w \cdot m^{w-1}}{m^w R^w + Z} > \frac{R^w}{m R^w + Z}$$

The inequality holds because $m^{w-1} > 1$ for $m \geq 2, w > 1$, and $m^{w-1}(mR^w + Z) > m^w R^w + Z$ reduces to $m^{w-1} Z > Z$. $\square$

**Counterexample.** Merging does NOT always help the *larger* party with unequal resources. With $R = [0.5, 0.3, 0.2]$ and $w = 2$, merging labs 0 and 2 (large + small):

| | Lab 0 | Lab 2 | Coalition |
|---|---|---|---|
| **Before** | 0.658 | 0.105 | 0.763 |
| **After** (equity 71%/29%) | 0.603 | 0.241 | 0.845 |

The coalition gains, but lab 0 **loses** (−0.054). Intuitively, lab 0 already had a disproportionate share from $w > 1$; proportional equity dilutes this advantage. Lab 0 would need to negotiate a larger equity share to benefit.

**Two-player case.** With only two players and no outside competition, merging is zero-sum — total $\Omega$ is always 1. The larger player always loses with proportional equity.

### Numerical examples (pure Tullock)

| Setup | Merging | All benefit? |
|---|---|---|
| 3 equal, $w=2$ | 2 of 3 | Yes |
| 3 unequal [.5,.3,.2], $w=2$ | 2 smaller | Yes |
| 3 unequal [.5,.3,.2], $w=2$ | large + small | **No** (large loses) |
| 5 players (default $R$), $w=2$ | top 3 | Yes (equal size) |
| 3 equal, $w=5$ | 2 of 3 | Yes |
| 3 equal, $w=1.1$ | 2 of 3 | Yes |
| 2 unequal, $w=2$ | both | **No** (large loses) |

## Full model with alignment and amity

In the full model, merging has three effects:

1. **Tullock effect**: The merged entity commands a larger $\Omega$ share (always positive for the coalition).
2. **Safety internalization**: The merged entity invests more in safety because it internalizes the externality among co-owners. Safety spending jumps dramatically (e.g., from $s \approx 0.02$–$0.05$ individually to $s \approx 0.15$–$0.22$ merged).
3. **Amity dilution**: Each co-owner's payoff from the merged entity uses equity-weighted amity $\sum_j e_j A_{ij}$, which can differ from their original self-amity of 1.

### Default parameters, merge OAI + Anthropic

Equity: OAI 50%, Ant 50% (equal resources).

| | Pre-payoff | Post-payoff | Change | Pre-$s$ | Post-$s$ |
|---|---|---|---|---|---|
| **OAI** | 0.370 | 0.441 | **+0.071** | 0.021 | 0.056 |
| **Ant** | 0.532 | 0.597 | **+0.064** | 0.045 | 0.056 |
| GDM | 0.559 | 0.564 | +0.005 | 0.054 | 0.063 |

Both merging parties gain. EHS improves from 0.821 to 0.876 (+0.055).

### Default parameters, merge top 3 (OAI + Ant + GDM)

Equity: 33.3% each (equal resources).

| | Pre-payoff | Post-payoff | Change | Pre-$s$ | Post-$s$ |
|---|---|---|---|---|---|
| **OAI** | 0.370 | 0.418 | **+0.048** | 0.021 | 0.217 |
| **Ant** | 0.532 | 0.615 | **+0.083** | 0.045 | 0.217 |
| **GDM** | 0.559 | 0.646 | **+0.087** | 0.054 | 0.217 |
| xAI | 0.175 | 0.190 | +0.015 | 0.064 | 0.092 |
| China | 0.262 | 0.286 | +0.023 | 0.164 | 0.175 |

All three merging parties gain. EHS improves from 0.821 to 0.921 (+0.101). The safety internalization effect is dramatic: safety spending goes from 2–5% to 22%.

### Selfish labs ($A = I$), merge top 3

| | Pre-payoff | Post-payoff | Change |
|---|---|---|---|
| **OAI** | 0.224 | 0.277 | **+0.053** |
| **Ant** | 0.224 | 0.277 | **+0.053** |
| **GDM** | 0.224 | 0.277 | **+0.053** |

EHS: 0.728 → 0.865 (+0.137). Largest EHS gain of any case, driven purely by safety internalization.

### 3-player cases ($R = [0.5, 0.3, 0.2]$, $w = 2$, $\delta = 0$, $\rho = 0$)

**Case 1: Selfish labs ($A = I$)**

| | Pre-payoff | Post-payoff | Change |
|---|---|---|---|
| **A** | 0.497 | 0.506 | **+0.009** |
| **B** | 0.186 | 0.304 | **+0.118** |
| C | 0.083 | 0.064 | −0.019 |

Coalition gains +0.127. Both merging parties benefit, even with unequal sizes. EHS: 0.766 → 0.874.

**Case 2: B is altruistic ($A_B = [0.8, 1.0, 0.8]$)**

| | Pre-payoff | Post-payoff | Change |
|---|---|---|---|
| **A** | 0.568 | 0.506 | **−0.062** |
| **B** | 0.678 | 0.762 | **+0.084** |
| C | 0.097 | 0.067 | −0.029 |

Coalition gains +0.022. B benefits but **A loses**. This is the one case where merging hurts a coalition member — but note the coalition as a whole still gains slightly, and A's loss is due to the size asymmetry (A has 0.5 vs B has 0.3), not the amity effect. EHS: 0.811 → 0.877.

*Compare with old analysis*: The previous version of this document claimed the coalition sum dropped from 1.294 to 0.854 (−0.440). That analysis was incorrect because it treated the merged entity as a single player whose payoff replaced both A's and B's payoffs, rather than splitting the merged entity's controlled resources back to shareholders via equity.

**Case 3: All altruistic ($A_{ij} = 0.8$ off-diagonal)**

| | Pre-payoff | Post-payoff | Change |
|---|---|---|---|
| **A** | 0.853 | 0.880 | **+0.027** |
| **B** | 0.772 | 0.835 | **+0.064** |
| C | 0.737 | 0.781 | +0.044 |

Coalition gains +0.091. Both parties benefit. EHS: 0.908 → 0.960. The "ecosystem diversity" concern from the old analysis does not materialize under equity shares: each shareholder still values the merged entity's aligned ASI proportionally, so the amity benefit is preserved rather than destroyed.

### Sweep over $w$ (merge top 3, default amity)

| $w$ | OAI Δ | Ant Δ | GDM Δ | Sum Δ | EHS Δ |
|---|---|---|---|---|---|
| 1.0 | +0.005 | +0.017 | +0.020 | +0.042 | +0.035 |
| 1.5 | +0.034 | +0.061 | +0.065 | +0.160 | +0.070 |
| 2.0 | +0.048 | +0.083 | +0.087 | +0.218 | +0.101 |

All parties gain at every $w$ tested (including $w = 1$ where the pure Tullock effect vanishes). The gains increase with $w$.

## Can every pair find a mutually beneficial equity split?

With proportional equity ($e_i = R_i / R_M$), the larger merging partner can lose when labs are unequal in size. But equity is negotiable. Does there always exist *some* equity split where both labs gain?

### Numerical evidence

We swept equity from 0.05 to 0.95 for all 10 lab pairs across 6 parameter regimes (default, low $k$, no public good/correlation, $z = 0$, $w = 1$, and extreme amity asymmetry). **In every single case, a zone of mutual benefit exists.** Some highlights:

| Pair | Parameters | Zone of $e_i$ | Best mutual gain |
|---|---|---|---|
| OAI + Ant | default | [0.40, 0.65] | +0.071, +0.064 |
| Ant + GDM | default | [0.20, 0.85] | +0.082, +0.082 |
| GDM + China | $w = 1$ | [0.71, 0.74] | +0.002, +0.006 |
| OAI + China | default | [0.75, 0.95] | +0.041, +0.044 |
| OAI (selfish) + Ant (altruistic 0.9) | default | [0.60, 0.95] | +0.057, +0.051 |
| Ant + GDM | low $k = 5$ | [0.05, 0.95] | +0.130, +0.134 |

The narrowest zone is GDM + China at $w = 1$ (only 4 percentage points wide), where the Tullock effect vanishes and safety internalization provides only marginal gains. But even here, a deal exists.

Cross-bloc mergers (US lab + China) require the US lab to take a larger equity share ($e \approx 0.75$–$0.85$) to compensate for losing amity value — the US lab has low amity toward China, so receiving Chinese-aligned equity is worth less.

### Analytical intuition

Three forces make mutually beneficial mergers ubiquitous:

1. **Coalition surplus is always positive.** In the pure Tullock contest, $(R_A + R_B)^w > R_A^w + R_B^w$ for $w > 1$, so the coalition always captures more total $\Omega$. This surplus can be redistributed via equity.

2. **Safety internalization creates additional surplus.** The merged entity internalizes the safety externality among co-owners, investing far more in safety (e.g., 2% → 22%). This raises alignment probability and thus payoffs for both parties. This effect operates even at $w = 1$.

3. **Equity is a continuous knob.** As $e_A$ increases from 0 to 1, lab A's payoff increases monotonically (more equity → more resources controlled), while lab B's decreases. By continuity, there exists an $e^*$ where both parties are better off than pre-merger, provided the total surplus is positive.

The argument in (3) is not a formal proof because the Nash equilibrium changes with equity (making the payoff functions potentially non-monotone). However, the numerical evidence is overwhelming: across 60 pair-parameter combinations, not a single counterexample emerged.

**Open question.** We conjecture that in the full model with $n \geq 3$ players and $w \geq 1$, there always exists an equity split making both merging parties (weakly) better off. A formal proof would require showing that the total coalition surplus is positive for some equity split, which involves bounding the loss from reducing two independent alignment draws to one against the gains from Tullock concentration and safety internalization.

## Key takeaways

1. **The standard Tullock merging result survives the equity-share model** when merging parties are of similar size and there is outside competition. Under the previous (incorrect) analysis that treated the merged entity as a single payoff, merging appeared to destroy "ecosystem diversity value." Under equity shares, each shareholder retains their proportional claim on the merged entity's aligned ASI, preserving the amity structure.

2. **The main threat to a merger is size asymmetry, not amity.** When a much larger lab merges with a smaller one, the larger lab may lose *at proportional equity* because it dilutes its disproportionate Tullock advantage. But the coalition surplus is still positive, so a negotiated equity split (giving the larger lab more than proportional equity) makes both parties better off.

3. **Safety internalization is the dominant benefit.** In the full model, the merged entity invests far more in safety (e.g., 2% → 22%), dramatically improving EHS. This safety benefit is present regardless of amity values.

4. **Every pair has a mutually beneficial merger** in all tested parameter regimes (60 combinations). The zone of agreement varies from 4pp ($w = 1$, cross-bloc) to 80pp (Ant + GDM at low $k$), but always exists.

5. **Even $w = 1$ benefits from merging** in the full model, purely through safety internalization. The Tullock contest effect ($w > 1$) amplifies the benefit but is not necessary.
