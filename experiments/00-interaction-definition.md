# Model-Consistent Definition of Interaction (with LaTeX)

We work on an \(I \times J\) dose grid with transformed responses \(Z_{ij}\) (e.g., logit-viability, log-inhibition, etc.). Let \(w_{ij} \ge 0\) be weights.

## Model classes (the geometry you test in)

### Monotone surfaces (isotonic class)
Let \(\mathcal{M}\) be the set of 2D monotone surfaces (choose direction based on response definition):

- **Monotone decreasing** (typical for viability):  
  \[
  \theta_{i+1,j} \le \theta_{i,j},\quad \theta_{i,j+1} \le \theta_{i,j}\quad \forall i,j.
  \]
- **Monotone increasing** (typical for inhibition):  
  \[
  \theta_{i+1,j} \ge \theta_{i,j},\quad \theta_{i,j+1} \ge \theta_{i,j}\quad \forall i,j.
  \]

The **isotonic estimator** is a weighted least-squares projection onto \(\mathcal{M}\):
\[
\mathrm{Proj}_{\mathcal{M}}(Z)\;:=\;\arg\min_{\theta\in\mathcal{M}} \sum_{i=1}^I\sum_{j=1}^J w_{ij}\,\bigl(Z_{ij}-\theta_{ij}\bigr)^2.
\]

### Monotone-additive surfaces (null class)
Let \(\mathcal{A}\subset \mathcal{M}\) be the monotone-additive class:
\[
\theta_{ij} \;=\; \alpha + u_i + v_j,
\]
with monotonicity constraints on \(u\) and \(v\) (same direction as \(\mathcal{M}\)) and identifiability constraints (e.g., \(\sum_i u_i=0\), \(\sum_j v_j=0\)).

The **additive estimator** is the weighted least-squares projection onto \(\mathcal{A}\):
\[
\mathrm{Proj}_{\mathcal{A}}(Z)\;:=\;\arg\min_{\theta\in\mathcal{A}} \sum_{i=1}^I\sum_{j=1}^J w_{ij}\,\bigl(Z_{ij}-\theta_{ij}\bigr)^2.
\]

## A model-consistent definition of “interaction”

The key principle is:

> **Define interaction as the non-additive component in the same model class you test.**

### Null (no interaction)
Choose a **noise-free** monotone-additive surface in the transformed scale:
\[
\theta^{\text{add,true}} \in \mathcal{A}.
\]
Then generate observations as
\[
Z_{ij} \;=\; \theta^{\text{add,true}}_{ij} + \varepsilon_{ij}.
\]

### Alternative (interaction exists)
Choose a **noise-free** monotone surface that is *not* additive:
\[
\theta^{\text{iso,true}} \in \mathcal{M}\setminus \mathcal{A}.
\]
Define the **model-consistent true interaction** as the part of \(\theta^{\text{iso,true}}\) that cannot be explained by the additive class:
\[
\delta^{\text{true}}
\;:=\;
\theta^{\text{iso,true}} - \mathrm{Proj}_{\mathcal{A}}\!\bigl(\theta^{\text{iso,true}}\bigr).
\]
Then generate observations as
\[
Z_{ij} \;=\; \theta^{\text{iso,true}}_{ij} + \varepsilon_{ij}.
\]

### What this guarantees
1. **Monotonicity of the total surface**: \(\theta^{\text{iso,true}} \in \mathcal{M}\) by construction.  
2. **Interaction defined in the same geometry as the method**: \(\delta^{\text{true}}\) is defined via the same constrained projection \(\mathrm{Proj}_{\mathcal{A}}\) used by the analysis pipeline.

This avoids “mismatch” situations where you simulate an interaction shape that is outside the monotone model class (and then the isotonic fit must first repair monotonicity, confounding interaction with monotonicity correction).

## What does “synergy exists” mean? (choose and state one)

Because interaction is fundamentally a **landscape** \(\delta_{ij}\), “synergy” can be defined locally or summarized globally. Common, explicit choices are:

### (A) Pointwise synergy landscape
Report the full surface:
\[
\delta_{ij} \quad \text{for all }(i,j),
\]
with uncertainty (e.g., bootstrap intervals/bands).

### (B) “Any synergy anywhere” (extremal or area-based)
For viability-type synergy defined as a *dip* (more effect than additive), use for example:
\[
T_{\max} \;=\; \max_{i,j}\bigl(-\delta_{ij}\bigr),
\]
or an exceedance area:
\[
T_{\text{area}}(c)\;=\;\sum_{i,j} \mathbf{1}\{\delta_{ij} < -c\}.
\]

### (C) Overall interaction energy
A stable global summary is weighted energy:
\[
S_2 \;=\; \sum_{i,j} w_{ij}\,\delta_{ij}^2.
\]

**Important:** \(S_2\) is a *two-sided* “interaction exists” summary. Like an \(F\)-statistic in regression/ANOVA, it measures **magnitude** of deviation from the null model, not its **direction**. Direction (synergy vs antagonism) lives in the *sign pattern* of \(\delta_{ij}\).

In finite samples, it is often helpful to use a **studentized** (noise-normalized) version for hypothesis testing:
\[
S_{2,\text{norm}} \;=\; \frac{S_2}{\mathrm{SSE}_{\text{add}}},
\qquad
\mathrm{SSE}_{\text{add}} \;=\; \sum_{i,j} w_{ij}\,(Z_{ij}-\hat\theta^{\text{add}}_{ij})^2.
\]
This keeps the scientific meaning (“interaction energy”) but reduces sensitivity to the (unknown) noise scale, improving bootstrap calibration on small grids.

### (D) Directional (signed) summaries
For viability-type synergy (dip) vs antagonism (bump), define:
\[
S_{-} \;=\; \sum_{i,j} w_{ij}\,\max(-\delta_{ij},0),
\qquad
S_{+} \;=\; \sum_{i,j} w_{ij}\,\max(\delta_{ij},0).
\]
Interpretation depends on the response/transform direction; you must state which sign corresponds to “synergy” for your \(Z\)-scale definition.

#### A single signed “direction index” (optional)
If you want a **one-number** summary that indicates whether the interaction is *mostly* synergy-like or antagonism-like, you need a direction notion that cannot be forced to cancel.

A subtle but important fact in our setting is that both \(\hat\theta^{\text{iso}}\) and \(\hat\theta^{\text{add}}\) are **translation-invariant projections** (adding a constant surface remains feasible), so they preserve the weighted mean of \(Z\). Therefore,
\[
\sum_{i,j} w_{ij}\,\delta_{ij} \;=\; 0,
\]
which implies the total positive and negative *masses* satisfy \(\sum w\max(\delta,0)=\sum w\max(-\delta,0)\). So an \(L^1\)-mass difference cannot encode direction.

Instead, compare the **directional energies**:
\[
S^2_{-} \;=\; \sum_{i,j} w_{ij}\,\max(-\delta_{ij},0)^2,
\qquad
S^2_{+} \;=\; \sum_{i,j} w_{ij}\,\max(\delta_{ij},0)^2.
\]
Let \(S^2_{\text{syn}}\) be the energy in the *synergy* direction (e.g., \(S^2_{-}\) for viability on a decreasing \(Z\)-scale), and \(S^2_{\text{ant}}\) the energy in the opposite direction. Define the bounded index
\[
I_{\text{syn}}
\;=\;
\frac{S^2_{\text{syn}} - S^2_{\text{ant}}}{S^2_{\text{syn}} + S^2_{\text{ant}}}.
\]
Then \(I_{\text{syn}}\in[-1,1]\): values near \(+1\) indicate “mostly synergy”, values near \(-1\) indicate “mostly antagonism”, and values near \(0\) indicate **mixed-sign interaction** (both synergy-like and antagonism-like regions).

We recommend using \(I_{\text{syn}}\) as a *descriptive* direction summary and keeping \(S_2\) (or \(S_{2,\text{norm}}\)) as the primary global “interaction exists” test, because sign can legitimately vary across the dose grid.

## Reporting and normalization (important for interpretability)

Many statistics above (including \(S_2, S_+, S_-\) and weighted SSEs) are **weighted sums**. Their absolute magnitude therefore depends on the scale of the weights \(w_{ij}\), which can vary dramatically across datasets or preprocessing choices.

Example: if there are no replicates and you use a small variance floor \(\tau\), a common default is \(w_{ij}\approx 1/\tau\), which can make all weighted-sum statistics numerically large even when the underlying interaction amplitudes \(\delta_{ij}\) are modest.

A simple, comparable set of *normalized* effect sizes divides by total weight:
\[
\bar S_2 \;=\; \frac{S_2}{\sum_{i,j} w_{ij}},\qquad
\bar S_{+} \;=\; \frac{S_{+}}{\sum_{i,j} w_{ij}},\qquad
\bar S_{-} \;=\; \frac{S_{-}}{\sum_{i,j} w_{ij}}.
\]
These are weighted *averages* (average squared interaction, and average positive/negative deviation), which are easier to compare across matrices.

If your primary goal is **inference** (a calibrated p-value) rather than effect-size comparability across matrices, studentization by \(\mathrm{SSE}_{\text{add}}\) is often the more relevant normalization:
\[
S_{2,\text{norm}} \;=\; \frac{S_2}{\mathrm{SSE}_{\text{add}}}.
\]

Similarly, for weighted residual sums of squares,
\[
\mathrm{MSE}_{\text{iso}} \;=\; \frac{\mathrm{SSE}_{\text{iso}}}{\sum w},\qquad
\mathrm{MSE}_{\text{add}} \;=\; \frac{\mathrm{SSE}_{\text{add}}}{\sum w},\qquad
\bar T_{\text{int}} \;=\; \frac{\mathrm{SSE}_{\text{add}}-\mathrm{SSE}_{\text{iso}}}{\sum w}.
\]

Note that statistics like \(\max(-\delta)\) do **not** depend on \(w\) and are already scale-stable.

## Why \( \mathrm{SSE}_{\text{add}} - \mathrm{SSE}_{\text{iso}} \) can be misleading in toy examples

The statistic
\[
T_{\text{int}} \;=\; \mathrm{SSE}_{\text{add}} - \mathrm{SSE}_{\text{iso}}
\]
often reflects **how well the isotonic model can chase noise under monotonicity**, not only true interaction.

If the underlying surface is strongly monotone and the noise is modest, the isotonic fit can become a near-saturated smoother (sometimes nearly interpolating), making \(\mathrm{SSE}_{\text{iso}}\) small under *both* null and alternative. Then \(T_{\text{int}}\) may not separate scenarios cleanly.

In that regime, statistics built directly from the inferred interaction surface \(\delta\) (e.g., \(S_2\), \(S_{-}\), \(T_{\max}\)) tend to align better with the scientific story: **interaction is a structured deviation from monotone additivity**.
