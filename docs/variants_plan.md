The challenge of exploring all these variants is that many of them overlap. By default we want just one at a time, line plots with survival probability and average safety investment. For each plot read it and ONLY after reading it write a description comparing it to the original case.


- For the safety elasticity parameter c / (k s^α + c), pick parameters such that
1% spending -> 20% misalignment, 50% spending -> 2% misalignment

Put all the plots in a report `report.ipynb` importing from `src/report/` and run it.

- more than two actors: one plot sweeping from 2 to 10 actors with default k, alpha
- actors have different starting resources
    - with 2 actors: ratio of resources on x axis, default k
    - with 5 actors: resources distributed 1/i
- one of them (A) has a comparative advantage in safety vs the other (B)
    - assume that a's safety research is twice as effective (double the k) as the b's. this plot should also have survival probability for A's AI and B's AI separately
- safety research is a public good
    - 5 actors, default k and alpha, but the s that goes into the formula is s_self^delta + s_total^(1-delta)
- winner-take-all dynamics c^w
    - just 2 actors, default k and alpha, sweep w from 0 to infinity
- alignment of different AIs is correlated
    - think about an appropriate model for this which is
        - parameterized by r from 0 to 1 
        - 0 -> uncorrelated case
        - 1 -> perfectly correlated (all AIs either aligned or misaligned if actors spend same safety ratio)
        - natural way to do this would be assuming that each ai has a critical safety ratio sampled such that c / (k s^α + c) holds. they are correlated via a reasonable copula. document which one you chose. also tell me if this is intractable to compute the nash for
    - sweep over r from 0 to 1 with default parameters
- two more plots of claude's choice that examine interactions between parameters
- three more plots of claude's choice based on whatever are maximally interesting to claude