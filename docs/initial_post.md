# AI Race Game Theory: Initial Model

Consider this toy game-theoretic model of AI safety: The US and China each have an equal pool of resources they can spend on capabilities and safety. Each country produces either AIs aligned to it or misaligned AIs. If one country's AI is misaligned, both payoffs are zero. If both AIs are aligned, a payoff of 100 is split proportional to the resources invested in capability.

The ratio of safety to capabilities investment determines the probability of misalignment; specifically, probability of misalignment is c / (ks + c) for some "safety effectiveness parameter" k. (At k=1 we need a 1:1 safety:capabilities ratio to have a 50% chance of alignment per AI, whereas at k=100 we only need a 1:100 ratio.) There is no coordination, so we want the Nash equilibrium. What's the equilibrium strategy in terms of k, and what is the outcome? It turns out that:

By symmetry, both countries will spend an equal fraction s* on safety, making the probability of alignment each P and so the overall probability of humanity surviving P^2. The payout to each country is thus 50P^2.

When safety is ineffective (low k): Countries invest heavily in safety (s* approximately 92% at k=0.1), but survival probability is still low (P^2 approximately 29%) because safety just doesn't work well.

When safety is effective (high k): Countries actually invest less in safety (s* approximately 33% at k=10), yet survival probability is higher (P^2 approximately 69%) because each unit of safety investment is more powerful.

As k increases, countries rationally sacrifice safety for competitive advantage. Even though safety becomes more effective, the survival probability only increases slowly with k. We need k>100 for a survival probability over 90%.

It follows that actors having perfect knowledge of consequences and acting selfishly is not sufficient to drive x-risk much below 10%, unless we get alignment by default. The world either needs deliberate coordination, or a perception that x-risk is worse than losing the AI race.

## Variants to Explore

- More than two actors (partially explored in Armstrong et al. (2013))
- Actors have different starting resources, or one of them has a comparative advantage in safety
- Safety research is a public good
- Winner-take-all dynamics (payouts go as c^w for some w>1)
- Safety elasticity parameter (misalignment probability is c/(ks^alpha + c))
- Alignment of different AIs is correlated
- Discrete choices rather than continuous (partially explored in the recent RAND report "A Prisoner's Dilemma in the Race to Artificial General Intelligence")
