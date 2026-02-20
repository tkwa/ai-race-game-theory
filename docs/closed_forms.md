# Closed-Form Equilibrium Results

## Setup

We consider $n$ actors each choosing a safety fraction $s_i \in [0,1]$. The core primitives are:

- **Alignment probability:** $P(s) = \frac{k s^\alpha}{k s^\alpha + (1-s)}$
- **Capability share:** $\text{share}_i = \frac{(1-s_i)^w}{\sum_j (1-s_j)^w}$
- **Joint survival:** $\prod_j P(s_j)$ (independent case)
- **Payoff:** $U_i = \text{joint survival} \times 100 \times \text{share}_i$

## 1. General symmetric equilibrium ($\alpha = 1$, arbitrary $n$, $w$)

With $\alpha = 1$, we have $P(s) = \frac{ks}{ks + c}$ where $c = 1 - s$, and $P'(s) = \frac{k}{(ks+c)^2}$.

### Deriving the first-order condition

At a symmetric equilibrium, all actors play $s^*$. Actor $i$'s payoff, holding others fixed at $s^*$, is:

$$U_i = P(s_i) \cdot P(s^*)^{n-1} \cdot 100 \cdot \frac{(1-s_i)^w}{(1-s_i)^w + (n-1)(1-s^*)^w}$$

Writing $c_i = 1-s_i$, $c^* = 1-s^*$, and $g(s_i) = \frac{c_i^w}{c_i^w + (n-1){c^*}^w}$, we differentiate:

$$\frac{dU_i}{ds_i} = P^{*\,n-1} \cdot 100 \left[ P'(s_i)\, g(s_i) + P(s_i)\, g'(s_i) \right] = 0$$

We need $g'(s_i)$ evaluated at $s_i = s^*$. Since $g = c_i^w / (c_i^w + D)$ with $D = (n-1){c^*}^w$ constant:

$$\frac{dg}{ds_i} = -\frac{dg}{dc_i} = -\frac{w\, c_i^{w-1} D}{(c_i^w + D)^2}$$

At $c_i = c^*$: $g(s^*) = 1/n$ and $g'(s^*) = -\frac{w(n-1)}{n^2 c^*}$.

The FOC at the symmetric equilibrium is therefore:

$$\boxed{P'(s^*) = P(s^*) \cdot \frac{w(n-1)}{n\, c^*}}$$

### Solving for equilibrium alignment probability

Let $p = P(s^*)$. We can re-parameterize in terms of $p$:

$$s = \frac{p}{k(1-p) + p}, \quad c = \frac{k(1-p)}{k(1-p)+p}, \quad ks + c = \frac{k}{k(1-p)+p}$$

Then $P'(s) = \frac{(k(1-p)+p)^2}{k}$ and $\frac{P'}{P} = \frac{(k(1-p)+p)^2}{pk}$.

Substituting into the FOC:

$$\frac{(k(1-p)+p)^2}{pk} = \frac{w(n-1)}{n} \cdot \frac{k(1-p)+p}{k(1-p)}$$

Cancelling one factor of $k(1-p)+p$ and cross-multiplying:

$$n(1-p)\bigl[k(1-p) + p\bigr] = pw(n-1)$$

Expanding and collecting powers of $p$:

$$\boxed{n(k-1)\,p^2 \;-\; \bigl[n(2k-1) + w(n-1)\bigr]\,p \;+\; nk \;=\; 0}$$

This is a **quadratic in $p$**, with solution:

$$\boxed{p = \frac{n(2k-1) + w(n-1) - \sqrt{\Delta}}{2n(k-1)}}$$

where the discriminant is:

$$\Delta = \bigl(n - w(n-1)\bigr)^2 + 4nkw(n-1)$$

(We take the minus root to get $p < 1$.)

**Joint survival** is $p^n$.

### Proof that the discriminant is always positive

$\Delta = (n - w(n-1))^2 + 4nkw(n-1)$. The first term is non-negative, and since $k > 0$, $w > 0$, $n \ge 2$, the second term is strictly positive. So $\Delta > 0$ always.

### Special case: $w = 1$ (no winner-take-all)

$$n(k-1)\,p^2 - (2nk-1)\,p + nk = 0, \qquad \Delta = 1 + 4nk(n-1)$$

### Special case: $n = 2$, $w = 1$

$$2(k-1)\,p^2 - (4k-1)\,p + 2k = 0, \qquad \Delta = 1 + 8k$$

For example, at $k = 33.9$: $p \approx 0.897$, joint survival $\approx 0.805$.

### Large-$k$ asymptotics

For $k \gg 1$ with $w = 1$:

$$p \approx 1 - \sqrt{\frac{n-1}{nk}}, \qquad \text{joint survival} \approx 1 - \sqrt{\frac{n(n-1)}{k}}$$

For general $w$:

$$p \approx 1 - \sqrt{\frac{w(n-1)}{nk}}, \qquad \text{joint survival} \approx 1 - \sqrt{\frac{nw(n-1)}{k}}$$

These follow from expanding the quadratic formula to leading order in $1/k$.

### Equilibrium safety spending

From the re-parameterization, the equilibrium safety fraction is:

$$s^* = \frac{p}{k(1-p) + p}$$

where $p$ is the solution above. For large $k$:

$$s^* \approx \frac{1}{k} \left(1 + \sqrt{\frac{w(n-1)}{nk}}\right)^{-1} \approx \frac{1}{k}$$

So equilibrium safety spending is roughly $1/k$ regardless of $n$ and $w$ — actors spend very little on safety when safety technology is effective.

## 2. Public good safety ($\delta$) drops out in the symmetric case ($\alpha = 1$)

With public-good spillovers, actor $i$'s effective safety is $\hat{s}_i = s_i^\delta \cdot \bar{s}^{1-\delta}$ where $\bar{s} = \frac{1}{n}\sum_j s_j$.

At symmetric equilibrium $\hat{s}_i = s^*$ for all $i$, but the FOC changes because $\bar{s}$ depends on $s_i$. Actor $i$'s payoff involves $P(\hat{s}_j)$ for all $j$, and $\hat{s}_j$ depends on $s_i$ through $\bar{s}$.

Computing the total derivative of payoff w.r.t. $s_i$:

$$\frac{d\hat{s}_i}{ds_i}\bigg|_{s_i=s^*} = \delta + \frac{1-\delta}{n} = \frac{\delta(n-1)+1}{n}$$

$$\frac{d\hat{s}_j}{ds_i}\bigg|_{s_i=s^*} = \frac{1-\delta}{n} \quad (j \ne i)$$

The alignment-probability part of the FOC involves:

$$P'(s^*)\frac{d\hat{s}_i}{ds_i} + (n-1)P'(s^*)\frac{d\hat{s}_j}{ds_i} = P'(s^*)\left[\frac{\delta(n-1)+1}{n} + \frac{(n-1)(1-\delta)}{n}\right] = P'(s^*)$$

The sum telescopes to exactly $P'(s^*)$ — the $\delta$ terms cancel. Therefore **the FOC is identical to the $\delta = 1$ (purely private) case**, and the equilibrium is the same closed form as in Section 1.

**Intuition:** In the symmetric equilibrium, increasing your safety by $ds$ improves your own effective safety and (via the public good) everyone else's. But the total marginal effect on all alignment probabilities sums to the same value regardless of how the benefit is split between private and public channels.

**Caveat:** This only holds for the symmetric case with $\alpha = 1$. For $\alpha \ne 1$, the chain rule introduces $\alpha \hat{s}^{\alpha-1}$ factors that break the telescoping, so $\delta$ matters. It also matters in asymmetric equilibria.

## 3. General $\alpha$ — no closed form

For $\alpha \ne 1$, we have $P(s) = \frac{ks^\alpha}{ks^\alpha + (1-s)}$ and:

$$P'(s) = \frac{k s^{\alpha-1}(\alpha(1-s) + s)}{(ks^\alpha + (1-s))^2}$$

The symmetric FOC (with $w=1$) becomes:

$$\frac{\alpha(1-s) + s}{s(ks^\alpha + (1-s))} = \frac{(n-1)}{n(1-s)}$$

This is **transcendental in $s$** (due to the $s^\alpha$ term with irrational $\alpha$) and admits no closed-form solution.

However, we can characterize the solution qualitatively. For large $k$, the equilibrium $s^*$ is small and $P(s^*) \approx 1 - s^{*\,(\alpha-1)}/k$, so the survival probability is still close to 1. The numerical solver handles this efficiently.

## 4. Asymmetric resources ($\alpha = 1$, $w = 1$, 2 actors)

Actors have resources $R$ and $1-R$. Actor $i$'s capability share is $R_i c_i / (R_1 c_1 + R_2 c_2)$ where $c_i = 1-s_i$.

The FOC system is (for actor 1 with $k_1 = k_2 = k$):

$$\frac{1}{s_1(ks_1 + c_1)} = \frac{(1-R)\,c_2}{c_1\bigl(R\,c_1 + (1-R)\,c_2\bigr)}$$

with a symmetric equation for actor 2 (swapping $R \leftrightarrow 1-R$ and $1 \leftrightarrow 2$).

### Reduction to a single equation

Multiplying the two FOC equations:

$$\frac{1}{s_1 s_2 (ks_1+c_1)(ks_2+c_2)} = \frac{R(1-R)}{(Rc_1 + (1-R)c_2)^2}$$

Dividing them:

$$\frac{s_2(ks_2+c_2)}{s_1(ks_1+c_1)} = \frac{R\,c_1^2}{(1-R)\,c_2^2}$$

The second equation gives a relationship between $s_1$ and $s_2$. Substituting back yields a single polynomial equation, but it is degree 4+ in either variable and does not simplify to a readable closed form.

**At $R = 1/2$** (equal resources), symmetry gives $s_1 = s_2$ and we recover the closed form from Section 1.

## 5. Asymmetric safety technology ($k_A \ne k_B$)

With 2 actors having different $k$ values but equal resources, the FOC system is:

$$\frac{(k_A(1-p_A)+p_A)^2}{p_A k_A} = \frac{k_B(1-p_B)}{(k_B(1-p_B)+p_B)} \cdot \frac{1}{c_A + c_B}$$

and the corresponding equation for $B$. Here $c_i = k_i(1-p_i)/(k_i(1-p_i)+p_i)$.

This is a system of two coupled nonlinear equations in $(p_A, p_B)$. In principle, elimination yields a single polynomial in one variable, but it is high-degree and provides no insight beyond the numerical solution.

## 6. Correlated alignment (Gaussian copula)

With equicorrelation $\rho$, joint survival is $\Phi_n\bigl(\Phi^{-1}(p_1), \ldots, \Phi^{-1}(p_n);\, \rho\bigr)$, where $\Phi_n$ is the $n$-variate normal CDF with equicorrelation matrix.

The FOC involves $\partial \Phi_n / \partial p_i$, which can be expressed via the conditional normal density, but the result involves evaluating $\Phi_{n-1}$ (the $(n-1)$-variate CDF). There is **no elementary closed form** for the equilibrium — even the joint survival itself requires numerical integration for $n \ge 2$.

## Summary

| Case | Closed form? | Reference |
|------|:---:|---|
| Symmetric, $\alpha=1$, any $n$, any $w$ | **Yes** — quadratic in $p$ | Section 1 |
| + public good $\delta$ (symmetric, $\alpha=1$) | **Same** — $\delta$ drops out | Section 2 |
| General $\alpha$ | No — transcendental | Section 3 |
| Asymmetric resources ($\alpha=1$) | In principle, but degree 4+ | Section 4 |
| Asymmetric $k$ ($\alpha=1$) | In principle, but high-degree system | Section 5 |
| Correlated alignment | No — requires numerical CDF | Section 6 |
