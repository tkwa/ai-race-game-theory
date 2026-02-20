# Parameter Estimates

## Resource vector R

Resources reflect each lab's share of current AI capability, considering compute infrastructure, frontier model standing, funding, and talent. $\sum R_i = 1$.

| Player | R | Justification |
|--------|------|---------------|
| GDM | 0.30 | Alphabet capex $175-185B in 2026, proprietary TPUs (Ironwood/v7), self-funded from $300B+ revenue, co-frontier (Gemini 3.1 Pro, Feb 2026). Also a compute *provider* to Anthropic and others. |
| OAI | 0.25 | Co-frontier (GPT-5.2, GPT-5.3-Codex), $1.4T in committed cloud compute (Azure, Oracle, AWS), $730-800B valuation, $20B ARR. Largest startup revenue but depends on external cloud. |
| Ant | 0.15 | At or near frontier (Claude Opus 4.6, Feb 2026), $380B valuation, ~$14B ARR. Smallest headcount but highest talent retention (80%, winning talent war 8:1 from OAI, 11:1 from GDM). ~$80B cloud spend committed through 2029 (AWS + Google Cloud + Azure). |
| xAI | 0.15 | Largest single GPU cluster (555K GPUs, Memphis Colossus, ~$18B in silicon). But 2-4 months behind frontier (Grok 4.20 beta), smallest team, lowest revenue ($500M). Raw compute is high but model quality lags. |
| China | 0.15 | Aggregate of DeepSeek, Alibaba/Qwen, ByteDance, Baidu, etc. Strong algorithmic efficiency (DeepSeek R1/V3 near-frontier at low cost), leads open-weight development (63% of HuggingFace fine-tunes). But severely hardware-constrained: Huawei Ascend output ~5% of Nvidia's, 3090x HBM gap, SMIC stuck at 7nm. 2-6 months behind frontier depending on lab. |

### Key uncertainties in R

- **China's algorithmic efficiency vs. compute gap**: DeepSeek showed you can partially compensate for hardware with better algorithms. If this continues, China's effective R could be higher (0.20). If frontier models require brute-force scale, lower (0.10).
- **xAI's compute-to-capability conversion**: 555K GPUs is enormous but Grok models lag. If xAI closes the gap, R could be 0.20. If compute alone doesn't translate, could be 0.10.
- **GDM vs OAI**: Could be closer to 0.28/0.27 if you weight model quality over infrastructure.

## Amity matrix A

$A_{ij}$ = how much lab $i$ values lab $j$'s aligned ASI controlling resources. $A_{ii} = 1$.

```
          OAI   Ant   GDM   xAI   China
OAI        1    0.4   0.4   0.3    0
Ant       0.5    1    0.5   0.3    0.1
GDM       0.4   0.4    1    0.3    0
xAI       0.2   0.2   0.2    1     0
China     0.3   0.3   0.3   0.2    1
```

### Reasoning by row

**OpenAI (row)**: Moderate amity toward Anthropic and GDM (0.4) — commercial competitors but broadly aligned values, significant talent exchange. Lower toward xAI (0.3) — less trust in Musk's governance. Zero toward China — per spec, US labs average 0 toward China due to competitive salience.

**Anthropic (row)**: Highest amity of any US lab. Explicitly safety-motivated — has stated preference for any aligned ASI over misaligned ASI. 0.5 toward OAI and GDM reflects genuine (if qualified) preference for competitor success over extinction. 0.3 toward xAI — safety culture gap. Slight nonzero toward China (0.1) — Anthropic's safety-first framing suggests they'd weakly prefer Chinese-aligned ASI over misaligned ASI, even if they wouldn't say so publicly.

**Google DeepMind (row)**: Similar to OAI. 0.4 toward OAI and Anthropic. 0.3 toward xAI. Zero toward China.

**xAI (row)**: Lowest amity toward other US labs (0.2 each). Musk's combative, zero-sum framing toward competitors. Treats other labs as adversaries more than peers. Zero toward China.

**China (row)**: Higher amity toward US than reverse (0.2-0.3). Chinese leadership is pragmatic — US-aligned ASI is far preferable to misaligned ASI from China's perspective. Lower toward xAI (0.2) — Musk's relationship with US government and unpredictability. Relatively even across US labs otherwise.

### Key uncertainties in A

- **US-China amity**: Could argue US labs should have small positive amity toward China (0.05-0.1) if you weight safety researchers' views. Kept at 0 per spec to reflect dominant competitive framing.
- **xAI's amity**: Musk is volatile. Could be higher (0.3-0.4) if you think his competitive rhetoric doesn't reflect actual preferences. Could be even lower if you take the zero-sum framing literally.
- **Anthropic's amity toward China**: Most uncertain entry. 0.1 reflects a weak safety-motivated preference; could be 0 if competitive pressures dominate internally.
