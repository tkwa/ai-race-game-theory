# Multi-turn / temporal extension

## Dynamics not captured by the current one-shot model

### 1. Stochastic arrival (the big one)

The current model assumes all labs develop ASI simultaneously and power is distributed by Ω. In reality, labs arrive at different times. The first aligned ASI could prevent subsequent misaligned ones; the first misaligned ASI could cause immediate catastrophe.

A patent-race model (Loury 1979) would have each lab's hazard rate h_i ∝ c_i, with the first arrival determining the outcome. The probability of lab i arriving first is c_i / Σc_j — superficially similar to the current Ω shares. But the key difference: in the current model, a misaligned lab B threatens the world even if aligned lab A has higher capability. In a race model, if A arrives first and is aligned, B's misalignment never materializes.

This means the current model is **pessimistic** relative to a race model when the leading lab is likely aligned, and approximately correct when arrival times are close.

### 2. Adaptive strategies / repeated game

Labs observe competitors and adjust. If Anthropic visibly increases safety spending, others might free-ride harder (moral hazard, which we already see in the "tell everyone" intervention) or reciprocate (repeated-game cooperation). The current one-shot Nash can't capture punishment strategies or trust-building.

### 3. Endogenous timing

Safety investment doesn't just affect alignment probability — it slows you down. A lab spending 20% on safety arrives later than one spending 3%. The current model captures the capability penalty but not the timing penalty. In a race, arriving late means someone else's (possibly misaligned) AI already exists.

### 4. Information revelation

As labs develop AI, they learn about alignment difficulty (updating k). Early results from RLHF, interpretability, etc. shift everyone's beliefs. A lab that discovers alignment is harder than expected might increase safety spending — or panic-race to arrive before competitors who haven't gotten the bad news.

### 5. Regulatory feedback

Racing behavior triggers government intervention (which is itself endogenous). The USG slowdown intervention is static, but in reality regulation arrives in response to observed behavior.

## Proposed extension: two-stage stochastic race

The most impactful extension is a two-stage stochastic race:

- **Stage 1**: Each lab's probability of arriving first is proportional to c_i^w (same as current Ω).
- **If first arrival is aligned**: With probability q ("defense advantage"), the aligned ASI prevents all subsequent misaligned AI. With probability 1−q, the game continues to stage 2.
- **If first arrival is misaligned**: Catastrophe with probability z (same as current), otherwise continues.
- **Stage 2**: Remaining labs resolve as in the current simultaneous model.

This adds one parameter q (first-mover defense advantage) and changes the outcome calculation but reuses all existing Nash machinery. The computation would enumerate "who arrives first" (n cases) × "does defense hold" (2 cases) × remaining outcomes.

A fuller version would be continuous-time with Poisson arrivals and dynamic programming, but that's substantially harder to solve for Nash equilibrium (each lab optimizes an arrival-rate/safety tradeoff over a continuous strategy space, with the value function depending on the state of who has already arrived).

## Closest approximation in current model

The current model already captures several temporal effects through its parameters:

| Temporal dynamic | Current approximation | Gap |
|---|---|---|
| First-mover advantage | w (winner-take-all) | w captures power concentration but not timing |
| Defense advantage of aligned ASI | z (misaligned power) | z=0 ≈ "aligned ASI neutralizes threats" but applies simultaneously, not sequentially |
| Arrival order correlated with capability | Ω ∝ (R·c)^w | Correct in expectation, but doesn't model the discreteness of "first vs. second" |
| Correlated alignment difficulty | ρ (copula) | Correct — temporal discovery of shared difficulty is equivalent to correlation |
| Adaptive strategies | Not captured | One-shot Nash ≈ cooperative equilibrium from repeated play only if discount factor is high enough |

## Key diff between one-shot and race model

In the current model, lab B's misaligned AI threatens the world even if lab A is much more capable and aligned. In a race model with defense advantage, A arriving first and being aligned largely eliminates the threat from B. This means:

- Current model **overestimates** risk when the leading lab (GDM, with ~43% Ω share) is likely aligned
- Current model **underestimates** risk when a less-capable lab with low safety investment could arrive first through luck
- The race model would make **consolidation interventions** (merge top 3, double GDM) even more positive, because a dominant aligned first-mover protects everyone
- The race model would make **fragmentation interventions** (duplicate labs) even more negative, because more labs means higher chance a low-safety lab gets lucky and arrives first

## Implementation estimate

The two-stage extension is tractable — roughly a day of work. It would produce a natural comparison: "here's how much the race framing changes each intervention's value." The main implementation steps:

1. Add `race_model_payoffs(s_all, R, A, k, alpha, w, z, q, delta, rho)` that enumerates first-arrival scenarios
2. Add `race_model_nash(...)` using the same iterated best-response solver
3. Run the same interventions and sensitivity analysis under the race model
4. Compare one-shot vs race results side by side
