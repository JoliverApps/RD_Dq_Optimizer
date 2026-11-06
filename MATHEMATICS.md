# Mathematical model and calculations

RD EQD2 Optimizer fits the Resilience–Depletion shoulder displacement Dq by comparing the equivalent doses of selected radiotherapy prescriptions. The running parameter is q; its value at the selected minimum is reported as Dq.

## RD survival law

For an acute dose x, the cumulative hazard and surviving fraction are

$$
H(x)=k\left[x-\frac{r}{s}(1-e^{-sx})\right],
\qquad S(x)=e^{-H(x)}.
$$

The parameters obey k > 0, s > 0, and r < 1. Their classical descriptors are

$$
D_0=\frac1k,\qquad D_q=\frac rs,\qquad
\alpha=k(1-r),\qquad \beta=\frac{krs}{2}.
$$

These α and β are the coefficients of the strict low-dose expansion

$$
H(x)=\alpha x+\beta x^2+O(x^3).
$$

| Quantity | Units |
| --- | --- |
| Dose, q, Dq, D₀, α/β | Gy |
| r | Dimensionless |
| k, s, α | Gy⁻¹ |
| β | Gy⁻² |

The response follows from the sign of r:

| Response | r | β | Dq | α/β |
| --- | --- | --- | --- | --- |
| Standard shoulder | 0 < r < 1 | Positive | Positive | Positive |
| Single-hit linear law | 0 | 0 | 0 | Undefined |
| Inverted shoulder | r < 0 | Negative | Negative | Negative |

In all three cases α, k, and D₀ are positive. The derivative

$$
H'(x)=k(1-re^{-sx})>0
$$

ensures decreasing survival for increasing dose. The curvature H″(x) = krs exp(−sx) has the sign of r.

## Signed parameter reconstruction

Enter a positive α/β reference for the q ≥ 0 branch. Denote this input by a₊. For each nonzero candidate, the signed ratio a is selected as

$$
a(q)=\begin{cases}a_+,&q>0,\\-a_+,&q<0.\end{cases}
$$

The application then reconstructs the RD parameters from the signed values:

$$
r=\frac{\sqrt{q^2+2qa}-q}{a},
\qquad s=\frac rq.
$$

The square root is the nonnegative principal root. Both branches satisfy s > 0; positive q gives 0 < r < 1, and negative q gives r < 0. The identities

$$
a=\frac{2(1-r)}{rs},\qquad
\alpha D_0=1-r,\qquad
\beta=\frac{(1-\alpha D_0)^2}{2qD_0}
$$

retain the signs of the selected parameters. In particular, q < 0 gives β < 0 while α and D₀ remain positive.

For positive q, the expression for r is evaluated in the rationalized form

$$
r=\frac{2q}{\sqrt{q^2+2qa}+q}
$$

to avoid subtracting nearly equal numbers. Negative q uses the signed numerator directly. In both cases s is calculated as r/q.

### The single-hit case

At q = 0, the application evaluates the linear law directly:

$$
r=0,\qquad H(x)=kx,\qquad S(x)=e^{-kx}.
$$

Here β = 0 and α/β is undefined. The rate s cannot be determined from this response and is displayed as N/A. The α/β reference remains an input setting for the nonzero candidates.

For any fixed positive fraction dose, the nonzero branches approach this linear response continuously. Parameter reconstruction is skipped at zero, so the calculation never divides by q there.

## Fraction effects, BED, and EQD2

Define the fraction effect in dose units as

$$
F(d)=\frac{H(d)}k=d-q(1-e^{-sd}).
$$

For n equal fractions with the same response parameters, cumulative hazards add as $H_{\mathrm{course}}=nH(d)$. With α = k(1−r),

$$
\mathrm{BED}=\frac{nH(d)}\alpha
=\frac{nF(d)}{1-r}.
$$

Using the reference fraction size d₀ = 2 Gy, equality of cumulative hazards gives

$$
\mathrm{EQD2}=d_0n\frac{F(d)}{F(d_0)}.
$$

The reference effect F(d₀) is positive. The scale k cancels from both BED and EQD2, so neither calculation requires D₀ or k as an input. Their absolute values cannot be estimated from this optimization alone.

At q = 0, F(d) = d and

$$
\mathrm{BED}=\mathrm{EQD2}=nd=D.
$$

### Numerical evaluation

The calculations use `expm1(-t)` for exp(−t)−1. This retains precision when t = sd is small.

For negative q, F(d) = d + q·expm1(−sd) is a sum of positive terms. Direct evaluation is also used for positive q ≤ d/4, where the subtracted term is bounded by one quarter of d.

For larger positive q, the implementation uses

$$
g(t)=1-\frac{1-e^{-t}}t,
\qquad F(d)=(1-r)d+rd\,g(sd).
$$

The terms are positive and avoid cancellation in d−q(1−exp(−sd)). For t < 0.01,

$$
g(t)=\frac t2-\frac{t^2}6+\frac{t^3}{24}
-\frac{t^4}{120}+\frac{t^5}{720}+O(t^6)
$$

is used, including g(0) = 0. Outside that interval, g(t) is evaluated as 1 + expm1(−t)/t.

The normalization 1−r is evaluated as ars/2 on the positive branch to retain precision when r approaches one. For negative r it is evaluated directly as 1−r. Its value is one in the linear case.

## Optimization and statistics

Let Eᵢ(q) be the EQD2 of prescription i, with N selected prescriptions and arithmetic mean μ(q). The available objectives are

$$
\mathrm{SSE}(q)=\sum_i[E_i(q)-\mu(q)]^2,
\qquad
\mathrm{SD}(q)=\sqrt{\frac{\mathrm{SSE}(q)}{N-1}},
$$

and range(q) = max Eᵢ(q) − min Eᵢ(q). SSE and sample SD have the same minimizers for a fixed selected set.

The search evaluates the selected grid, including both interval endpoints and one zero when the interval spans it. An interior grid minimum is refined with a golden-section search within its neighboring grid cell. Each prescriptions set is optimized independently.

For a tolerance p%, the profile bounds delimit the connected interval around the selected minimum where the profile measure is at most (1+p/100) times its minimum. SSE and SD optimization use sample SD as the profile measure; range optimization uses range. A missing bound indicates no crossing before the search endpoint.

Relative dispersion is reported as RSD = 100·SD/μ for the positive RD equivalent doses. A least-squares line E = intercept + b·d describes the trend against dose per fraction. The trend measure is

$$
\eta=100\,\frac{|b|(d_{\max}-d_{\min})}{\mu}.
$$

When all fraction doses are identical, the fitted slope and η are undefined. The analysis window also reports the 95% Student-t interval for the arithmetic mean,

$$
\mu\pm t_{0.975,N-1}\frac{\mathrm{SD}}{\sqrt N}.
$$

These quantities describe the selected prescription distribution.

Leave-one-out analysis repeats the optimization with each prescription omitted. α/β sensitivity analysis repeats it for each entered positive reference value, applying the q-dependent sign during every search.

## LQ comparison

The analysis window calculates comparison values from the selected reference a:

$$
\mathrm{BED}_{\mathrm{LQ}}=nd\left(1+\frac da\right),
\qquad
\mathrm{EQD2}_{\mathrm{LQ}}=nd\frac{a+d}{a+d_0}.
$$

At q = 0, this comparison uses the entered reference value, while the RD result is the linear-law limit. LQ comparison values do not enter the RD optimization objective. For negative a, LQ extrapolations can become negative; at a = −d₀, LQ EQD2 is undefined and is displayed as N/A.

## Implementation reference

The equations are implemented in [rd_eqd2_optimizer.py](rd_eqd2_optimizer.py).

| Function | Calculation |
| --- | --- |
| `_branch_alpha_beta` | Signed reference selection |
| `_signed_rd_parameters` | Reconstruction of a, r, and s |
| `_initial_hazard_factor` | α/k = 1−r |
| `_shoulder_weight` | Stable evaluation of g(sd) |
| `_fraction_effect_values` | F(d) = H(d)/k |
| `fraction_bed` | BED per fraction |
| `EQD2_FN`, `evaluate` | Prescription EQD2 values |
| `analyze` | Objective search, statistics, and profile bounds |
