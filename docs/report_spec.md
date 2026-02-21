# Model description

- 5 players OAI, Ant, GDM, xAI, China in a one-turn simultaneous game.
- Each has resources $R_i$ to spend on safety or capabilities, and wants to develop ASI aligned to them. The total of world resources $\sum R_i$ is set to $1$.
- Payouts for lab $i$ are a weighted average of $\Omega_j \tilde A_{ij}$, where $\Omega_j$ roughly tracks the expected share of the universe controlled by lab j's AI after the singularity. $\Omega_j$ can be interpreted either as $j$'s probability of dominating a unipolar future, or $j$'s share of resources controlled in a multipolar future.
  - $\tilde A_{ij} = 0$ if AI $j$ is misaligned.
  - $\tilde A_{ij} = A_{ij}$ if AI $j$ is aligned: The amity matrix A determines how much each lab values ASI being aligned to other labs, with 0 being equivalent to misaligned AI and 1 as good as developing ASI themselves.
    - $A_{ii} = 1$, so a lab gets payout 1 if its own aligned AI dominates the future.
- Post-singularity power $\Omega_i$ is split between labs proportional to $c_i^w$
  - $w$ determines how winner-take-all the AI race is. At $w=0$ all labs get an equal share regardless of capabilities investment, and $w=\infty$ the winner gets 100% of the lightcone
- Alignment probability of lab i goes as $1/(1 + \frac{c_i}{k S_i^\epsilon})$
  - $S_i := \left( \sum_j s_j \right)^\delta s_i^{1-\delta}$ is the company's effective safety level
  - $k$ tracks the ease of safety
  - $\epsilon$ tracks the elasticity of safety output to safety investment; whether there are diminishing or increasing returns.
  - $\delta$ tracks whether safety is a public good; at $\delta=0$ safety level of a lab depends entirely on their own investment; at $\delta=1$ it depends on aggregate investment.
- Alignment correlation
  - $r$ tracks how strongly alignment is correlated between labs.
    - At $r=0$ (independent draws): The required alignment threshold is drawn independently for each lab.
    - $r=1$: Alignment is equally difficult for everyone; each lab must clear a global threshold.
    - By default a Gaussian copula is used, but other choices are possible.
- $z$ determines how much power advantage a misaligned AI has over aligned competitors, via its effect on $\Omega_j$.
  - For aligned lab $j$: $\Omega_j \propto (R_j c_j)^w$.
  - For misaligned lab $j$: $\Omega_j \propto (R_j c_j)^{w(1-z)}$.
  - At $z=0$, misaligned AI competes on equal footing with aligned AI (the [strategy-stealing assumption](https://www.lesswrong.com/posts/nRAMpjnb6Z4Qv3imF/the-strategy-stealing-assumption) holds): $\Omega_j$ is the same whether or not lab $j$'s AI is aligned.
  - At $z=1$, any misaligned AI has $\Omega_j \propto 1$ regardless of the lab's actual investment — combined with winner-take-all dynamics ($w$), this approaches total takeover.
  - A lab with zero resources ($R_j = 0$) has $\Omega_j = 0$ for any $z < 1$.

We assume there is no cooperation, and labs maximize expected utility, so we want the Nash equilibrium.

### Parameter values

- $R$ and $A$ use Tao's estimates (see `docs/tao_params.md`). R = [OAI: 0.27, Ant: 0.27, GDM: 0.27, xAI: 0.09, China: 0.10]. See `docs/parameters.md` for Claude's earlier estimates and Tao's rationale for the changes.

- $R$: For the sensitivity analysis, alter China's resources to be in [0.05, 0.25], rescaling the rest to add up to 1.
- $A$: For the sensitivity analysis, shrink the off-diagonal entries towards 1 by up to 2x or dilate them away by up to 1.5x (making some negative). The label should be average amity [not counting diagonal ofc]
- $w$: 2.0 by default, range [1.0, 5.0]
- $k$, $\epsilon$: set them such that 1% spending -> 20% misalignment, 50% spending -> 2% misalignment. Doesn't make sense to vary because these are different worldviews.
- $\delta$: 0.5 by default, range [0, 0.75]
- $r$: 0.5 by default, range [0, 1]
- $z$: 0.9 by default, range [0.5, 1]

### Interpretion of resources and competition

Resources $R_i$ is defined as the lab's share of current resources (talent and compute).

As for competition, we define payouts as subjective utility, which don't necessarily correspond to the lab's final post-ASI power and should incorporate biases. But they could roughly correspond to:
- Expected share of lightcone
- p(controlling the universe) (in a nondeterminstic, unipolar world)

Biases like loss aversion and irrational racing could be incorporated into A and w.

## Report format and presentation

The `report.ipynb` should be an internal report with the following sections:
- "Toy model" explaining the basic 2-actor model described in `initial_post.md` and extending it to the most general subcase of the model that still has an easily understandable closed form (likely n actors, equal resources).
  - Plot: as in `initial_post.md`
  - Explanation of closed form, including `1 − √(n(n−1)/k)` with n actors. Explain that misalignment probability only decreases with k.
  - Briefly mention the closed form with n actors and general $w$, and how winner-take-all dynamics.
  - Plot: asymmetric resources (2 players), safety spending and joint survival vs resource ratio. X-axis from 0.5 to 1.
- "Full model" section explaining the model above
  - Explain that "Survival probability" can be instead interpreted as "expected human share" in multipolar post-ASI futures.
  - Plot for outcomes vs k, with default value of $\epsilon$ and all other params
  - Clustered bar graph showing, for each lab, their AI's share | alignment in black, and their AI's p(misalignment) in red
  - Table for interventions. Should be a bar graph of survival probability relative to the default case (shown as a vertical line). Both the absolute and relative probabilities for each case should be printed.
    - Remove China
    - Remove Anthropic
    - Duplicate every lab
    - Secretly make the largest lab 100% safe
    - Make the largest lab 100% safe, and tell only them
    - Make the largest lab 100% safe, and tell everyone
    - Double the largest lab's resources
    - Increase amity 10% of the way towards 1
    - Make safety a public good ($\delta = 1$)
    - Make the top 3 labs have mutual amity 1
    - Merge the top 3 labs (amity towards remaining labs set to average of the 3 rows)
    - Increase US -> China amity to China -> US levels
    - USG demands slowdown: forcibly reduce the capabilities investment of all the US labs by 33%; keep absolute safety level the same; China adapts
    - USG demands safety investment: Labs must spend at least 10% on safety
    - Set GDM's amity towards every other lab to 0
  - Sensitivity analysis
    - Similar graph format to [takeoff model](https://github.com/tkwa/ai-takeoff-model/). Sample from distributions of parameters, vary one at a time, subplots of expected human win% marginalized on each parameter with a range.
    - To allow easy comparison, y axis grid lines should be 0.02 in each subplot.

There should also be the following appendices:
- A writeup of methodology behind Claude's parameter choices
- When is computing the Nash equilibrium easy? Pull from subreport `closed_forms.md`
- Why Gaussian copula and what would a different copula change? By default just write what Claude is confident about without experiments, but could execute `plans/copula.md` in the future.

All plots should be saved to files in `plots/` so that the human can reference them in the post `README.md`.

The post will be based on the internal report, It will have a similar structure to https://github.com/tkwa/ai-takeoff-model/.
- It will only include some of the plots
- There will be (mostly) human-written sections for Introduction, Model, Graphs, Discussion, etc.
- The Model Description section of this document will be adapted to the post.
- Model Description has a subsection synthesizing Claude's and the human's parameter choices.

## Implementation details

- The tolerance in the Nash solver should be at MOST 1e-5 globally except for 