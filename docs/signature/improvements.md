**Signature Improvements — Implementation Notes**

*Based on Gyurkó, Lyons, Kontkowski & Field (2014) arXiv:1307.7244 • Chevyrev & Kormilitzin (2016) arXiv:1603.03788 • Morrill et al. (2020) arXiv:2006.00873*

**1. Per-channel standardisation before the signature**

Scale every embedding channel so

$$
\operatorname{StDev}(\Delta \text{channel}) = 1
$$

across the window before calling pysiglib. Without this, channels on different scales (e.g. raw price vs. a 0–1 volume ratio) dominate the signature terms numerically. Degree-$k$ iterated integrals scale as $L^k$ in path length, so unequal scales compound badly at depth 3–4 — the terms you care most about become noise relative to the large-scale channels.

**Ref:** *Gyurkó et al. §3.1 (C_p, C_s per-stream normalisation); Chevyrev & Kormilitzin §3.4*

**2. Increase truncation depth to 4**

The paper’s LASSO consistently identifies 4th-order iterated integrals as the most informative features across all three experiments. Not a single one of the top-15 selected terms is of order $\leq 3$. Operating at depth 3 means stopping one level before the signal starts.

**Ref:** *Gyurkó et al. §4, Tables 1–3 (all top multi-indices are length 4); Morrill et al. §5*

**3. Basepoint augmentation**

Prepend a fixed zero vector to the start of every path window. Without this, the signature is translation-invariant: two windows with identical relative dynamics but different absolute price levels produce identical feature vectors. The basepoint anchors the path to a known origin so that starting level is encoded.

**Ref:** *Chevyrev & Kormilitzin §3.2; Bonnier, Kidger, Perez Arribas, Salvi & Lyons (2019) NeurIPS*

**4. Factorial rescaling of terms**

Divide each degree-$k$ term by $k!$ before concatenating into the observation vector. For a linear path,

$$
X^I_{s,t} = \frac{(t - s)^k}{k!},
$$

so higher-degree terms decay factorially. Degree-3 and degree-4 terms can be two or more orders of magnitude smaller than degree-1 terms. Without rescaling, gradients are dominated by low-order terms, suppressing the 4th-order signal you are trying to use.

**Ref:** *Chevyrev & Kormilitzin §3.3; Reizenstein (2018) iisignature/esig library documentation*

**5. Add normalised cumulative volume as a path channel**

Include

$$
c_t = \frac{C_t}{C_{t_N}}
$$

as a coordinate in the embedded path. This lets the signature encode volume profile shape: whether trading is front-loaded, back-loaded, or uniform. The top 3 LASSO features in Experiment 1 all index this channel. Without it, two windows with identical price paths but opposite volume profiles are indistinguishable.

**Ref:** *Gyurkó et al. §3.1 channel c_t; §4.1 features (1,5,1,5), (5,1,5,1), (1,5,5,1)*

**6. Partial lead-lag transform to capture realised variance**

Construct a $(d+1)$-dimensional stream: all $d$ channels as ‘lead’, plus one extra channel that is the ‘lag’ of log-price only. The signed area between lead-price and lag-price equals the realised quadratic variation — invisible to the plain signature. The single most informative term in the paper, $(5,1,6,2)$, is precisely this interaction between cumulative volume and the lead-lag price pair.

**Ref:** *Gyurkó et al. §2.5 quadratic variation, §3.2–3.3 lead-lag construction; Figure 4*

**7. High–Low range as an intraday spread proxy**

Add

$$
\frac{H_t - L_t}{\mathrm{Close}_t}
$$

as a path channel. This acts as a low-frequency proxy for the bid-ask spread and realised volatility — the role of the spread channel $s_t$ in the paper, which required level-one order book data. Several top LASSO features in Experiment 2 involve spread-indexed terms, showing that spread dynamics carry information independent of price.

**Ref:** *Gyurkó et al. §3.1 spread channel s_t; Roll (1984) JF implicit spread; Parkinson (1980) extreme-value variance*

**8. Candlestick (OHLC) intrabar micro-path**

Instead of one Close point per bar, construct a 4-point intrabar path: O → H → L → C (or direction-adjusted). This encodes wick lengths, shadow ratios, and body position without needing higher-frequency data. The signature of this richer path at the same depth captures strictly more information because the intermediate extremes leave a geometric trace in the iterated integrals.

**Ref:** *Gyurkó et al. §2.4 piecewise-linear interpolation; Cuchiero, Möller & Svaluto-Ferro (2023) OHLC signature pricing*

**9. Multi-scale sub-window signature concatenation**

Compute signatures over multiple overlapping sub-windows (e.g. last 8, 16, 32 bars of a 32-bar window) and concatenate. A single-window signature loses temporal locality — it cannot tell whether a pattern occurred at the start or end of the window. Sub-window concatenation recovers this, which matters especially for time-of-day-dependent intraday patterns.

**Ref:** *Morrill et al. (2020) §3 window features; Fermanian (2021) Computational Statistics & Data Analysis §4*

**10. Agent inventory as a live path channel**

Include $position\_ratio \in [-1, 1]$ as a path coordinate at each timestep rather than appending it as a scalar after the signature. When inventory is part of the path, the signature captures higher-order interactions between market dynamics and the agent’s exposure history — e.g. whether variance was high while long, or whether the spread widened mid-position. These joint path-response correlations are what the rough-paths framework is built to encode.

**Ref:** *Levin, Lyons & Ni (2013) joint path-response signatures; Halperin (2020) QLBS joint state-action features*

