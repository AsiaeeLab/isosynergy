# Why we rescale residuals in the wild bootstrap (`residual_scale = "df"`)

This note explains the two knobs you see in the simulation reports:

- `boot_residual_scale <- "df"`
- `boot_df_null <- I + J - 3`

The short version is: **residuals from a fitted null model are “too small” because the model has absorbed some variation**, so the wild bootstrap needs a **degrees-of-freedom correction** to put the bootstrap noise back on the right scale.

---

## 1) Setup and notation

Work on an \(I\times J\) dose grid with transformed responses \(Z_{ij}\) and weights \(w_{ij}\ge 0\).
Let \(\mathcal A\) be the *monotone-additive* class (the null geometry):
\[
\theta_{ij}=\alpha + u_i + v_j,\qquad \theta\in\mathcal A,
\]
with monotonicity constraints on \(u\) and \(v\) and identifiability constraints (e.g. \(\sum_i u_i = 0\), \(\sum_j v_j=0\)).

Define the weighted projection (the null fit)
\[
\widehat\theta_{\text{add}}
\;=\;
\mathrm{Proj}_{\mathcal A}(Z)
\;:=\;
\arg\min_{\theta\in\mathcal A}\sum_{i,j} w_{ij}\,(Z_{ij}-\theta_{ij})^2.
\]
Let residuals under the null fit be
\[
r_{ij} \;=\; Z_{ij}-\widehat\theta_{\text{add},ij}.
\]

Let \(n_{\text{eff}}\) be the number of *effectively used* cells (e.g. those with finite \(Z_{ij}\) and \(w_{ij}>0\)); on a complete grid \(n_{\text{eff}}=IJ\).

---

## 2) Wild bootstrap under the additive null (what we want)

The “Rademacher wild bootstrap” constructs bootstrap pseudo-data
\[
Z_{ij}^\star
\;=\;
\widehat\theta_{\text{add},ij} + \xi_{ij}\,\tilde r_{ij},
\qquad
\xi_{ij}\in\{-1,+1\}\text{ i.i.d. with }\mathbb P(\xi_{ij}=1)=\tfrac12,
\]
and then refits the models on \(Z^\star\) to obtain a bootstrap distribution of the chosen statistic \(T^\star\).

The goal is that, **under the null**, the conditional distribution of \(T^\star\mid Z\) approximates the sampling distribution of \(T\).

---

## 3) Why the raw residuals \(r_{ij}\) are “too small”

Even under a perfect null, the fitted surface \(\widehat\theta_{\text{add}}\) is estimated from the same data \(Z\). This estimation *uses up degrees of freedom*, which shrinks residual variability.

To see the mechanism cleanly, consider the weighted linear model idealization
\[
z = X\beta + \varepsilon,\qquad \varepsilon\sim(0,\sigma^2 I),
\]
with weighted least squares and hat matrix \(H\). The residual vector is
\[
r = (I-H)\varepsilon.
\]
Then
\[
\begin{aligned}
\mathbb{E}\big[\lVert r\rVert_2^2\big]
&= \mathbb{E}\big[\varepsilon^\top (I-H)\varepsilon\big] \\
&= \sigma^2\,\operatorname{tr}(I-H) \\
&= \sigma^2\,(n-p),
\end{aligned}
\]
where \(p=\operatorname{tr}(H)\) equals the number of fitted degrees of freedom (for full-rank OLS/WLS, \(p=\operatorname{rank}(X)\)).

So the **average residual variance** is
\[
\frac{1}{n}\mathbb{E}\big[\lVert r\rVert_2^2\big]=
\sigma^2\Bigl(1-\frac{p}{n}\Bigr),
\]
which is *smaller than* \(\sigma^2\).

If we plug the *unadjusted* residuals into the wild bootstrap
\(\,Z^\star = \widehat z + \xi r\),
we are effectively bootstrapping with noise variance closer to
\(\sigma^2(1-p/n)\),
which makes the bootstrap distribution of many statistics **too concentrated** and can yield **anti-conservative** p-values (too many tiny p-values under the null).

The same idea applies to the weighted SSE:
with \(W=\mathrm{diag}(w)\), \(\mathrm{SSE}=\|W^{1/2}r\|_2^2\) and typically
\[
\mathbb{E}[\mathrm{SSE}] \approx (n_{\text{eff}}-p)\sigma^2
\quad\Rightarrow\quad
\widehat\sigma^2 \approx \frac{\mathrm{SSE}}{n_{\text{eff}}-p}
\;\text{ rather than }\;
\frac{\mathrm{SSE}}{n_{\text{eff}}}.
\]

---

## 4) The df-rescaling fix: \(\tilde r = r\cdot\sqrt{\tfrac{n}{n-p}}\)

The standard degrees-of-freedom correction is to inflate residuals by
\[
\tilde r_{ij}
\;=\;
r_{ij}\cdot
\sqrt{\frac{n_{\text{eff}}}{n_{\text{eff}}-p_{\text{null}}}},
\]
so that the bootstrap noise has (approximately) the right scale.

This is exactly what `residual_scale = "df"` implements in `R/bootstrap.R`:

- `df_null` plays the role of \(p_{\text{null}}\) (effective df used by the null fit),
- and the multiplier is \( \sqrt{n_{\text{eff}}/(n_{\text{eff}}-df_{\text{null}})} \).

---

## 5) What is \(p_{\text{null}}\) for a monotone-additive fit?

Our null fit is **not** an ordinary linear regression: it is a constrained QP (monotonicity + identifiability). That means the exact “hat matrix” and \(p=\operatorname{tr}(H)\) are not available in a simple closed form.

We use a pragmatic, model-aware approximation:

### (a) Unconstrained additive df (upper bound)

If we ignored monotonicity and only imposed identifiability, the additive surface has
\[
p_{\text{add,free}} = 1 + (I-1) + (J-1) = I+J-1
\]
free parameters (intercept plus \(I-1\) \(u\)-effects and \(J-1\) \(v\)-effects).

### (b) Monotone constraints reduce *effective* df via pooling

Under monotonicity, the fitted sequences \(u\) and \(v\) are typically **piecewise constant** (isotonic “blocks”).
Let
\[
K_u = \#\{\text{constant blocks in } \widehat u\},
\qquad
K_v = \#\{\text{constant blocks in } \widehat v\}.
\]
Heuristically, each block behaves like one free level, so the effective df is about
\[
p_{\text{null}} \;\approx\; 1 + K_u + K_v - 2 \;=\; K_u + K_v - 1.
\]
This is the rationale behind the “automatic” df estimate in `R/bootstrap.R` that counts distinct runs in \(\widehat u\) and \(\widehat v\).

---

## 6) Why a simple choice like \(I+J-3\) can make sense

Setting
\[
df_{\text{null}} = I + J - 3
\]
is a **stable rule-of-thumb** that corresponds to “one pooling event in each main effect”:

- “No pooling” would give \(K_u=I\), \(K_v=J\) \(\Rightarrow p_{\text{null}}=I+J-1\).
- “One tie/merge in \(u\)” gives \(K_u\approx I-1\).
- “One tie/merge in \(v\)” gives \(K_v\approx J-1\).

Plugging \(K_u\approx I-1\) and \(K_v\approx J-1\) into \(p_{\text{null}}\approx K_u+K_v-1\) yields
\[
p_{\text{null}} \approx (I-1) + (J-1) - 1 = I+J-3.
\]

Numerically this is a **mild** inflation. For example, on an \(8\times 8\) grid:
\[
n_{\text{eff}} = 64,\quad df_{\text{null}}=13
\quad\Rightarrow\quad
\sqrt{\frac{64}{64-13}} \approx 1.12.
\]
So we are not “changing the problem”; we are applying a small correction so that the bootstrap replicates reflect the noise level that remains after fitting the null.

In practice, we treat \(df_{\text{null}}\) as a **calibration knob**:
we choose a reasonable value (like \(I+J-3\)) and then *validate* it by checking that null p-values look roughly uniform in repeated-null simulations.

---

## 7) Related option: pointwise HC2 scaling (not just a global factor)

The code also supports a more local correction inspired by heteroskedasticity-robust bootstrap:
\[
\tilde r_{ij} = \frac{r_{ij}}{\sqrt{1-h_{ij}}},
\]
where \(h_{ij}\) is an approximate leverage from a (weighted) additive design.
This corresponds to `residual_scale = "hc2"` in `R/bootstrap.R`.

HC2 is often helpful when leverage varies strongly across cells; the df scaling is simpler and sometimes more stable on small grids.

---

## Takeaway

`residual_scale = "df"` implements the classic correction
\(\tilde r = r\sqrt{n/(n-p)}\)
so the wild bootstrap does not understate noise.

`df_null = I+J-3` is a pragmatic approximation of the null model’s effective degrees of freedom under monotone pooling (roughly “one merge in each main effect”), and we verify it empirically by checking null p-value calibration in simulation.
