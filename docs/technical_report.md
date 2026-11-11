# Reduced-order surrogate modelling for a parameterized steady Darcy-flow problem

## Sampling strategies, regression comparison, and POD-rank sensitivity

**Independent computational investigation**  
**Status:** preliminary research portfolio report; results are conditional on the stated numerical model and experimental designs.

---

## Abstract

This study investigates reduced-order surrogate modelling for a parameterized steady scalar Darcy-flow problem. A cell-centred finite-volume solver is used to generate pressure fields for a spatially heterogeneous permeability field controlled by a mean log-permeability, an amplitude, and an orientation parameter. The work contains two connected investigations. The first explores random, Latin-hypercube, Sobol, and uncertainty-driven adaptive sampling, compares a POD-Gaussian-process-regression (POD-GPR) surrogate with a direct neural-network baseline, and evaluates Sobol sensitivity indices for domain-averaged pressure. The second addresses a limitation of the initial model comparison by evaluating six regression families on a shared POD basis and common final training and test designs. A subsequent rank-sensitivity experiment holds the RBF-GPR family fixed while varying the number of retained POD modes and decomposing total pressure error into POD truncation and regression components.

In the controlled comparison at 256 training samples and 32 POD modes, RBF-GPR achieved a held-out global relative pressure error of approximately 0.100% and a relative flux error of approximately 1.168%, outperforming the five other tested regression families on these metrics. The pressure error was dominated by POD truncation at this rank. In the rank-sensitivity experiment, increasing the POD rank reduced the test pressure error until approximately rank 96, after which the error approached a regression-limited floor of roughly 1.57 × 10⁻⁴. These findings indicate that surrogate accuracy depends jointly on the reduced representation and the regression model. They are specific to the investigated solver, mesh, parameter bounds, and sampling designs; they do not establish a universally optimal regression method or POD rank.

## 1. Motivation and research questions

The purpose of this investigation is to explore a computational workflow for constructing and assessing reduced-order surrogates for parameterized flow simulations. The emphasis is not on claiming a new Darcy solver or demonstrating superiority over another research group's method. Instead, the study examines how sampling, regression, and reduced-basis dimension affect surrogate prediction accuracy and computational cost.

The investigation is organized around three questions:

1. How do different sampling designs perform when constructing surrogates for the parameterized Darcy problem?
2. When training data, POD representation, and final test cases are controlled, how do alternative regression families compare?
3. How does the retained POD rank affect truncation error, regression error, pressure prediction, and derived flux prediction?

The first question motivated the initial sampling study. The second and third arose from the need to isolate the sources of surrogate error more carefully.

## 2. Mathematical and numerical model

### 2.1 Governing equation and boundary conditions

The steady scalar Darcy problem is

\[
-\nabla\cdot\left(k(x,y)\nabla p(x,y)\right)=0,
\qquad (x,y)\in(0,1)^2,
\]

with Dirichlet pressure conditions

\[
p(0,y)=1,\qquad p(1,y)=0,
\]

and no-flow conditions on the horizontal boundaries,

\[
\boldsymbol{n}\cdot k\nabla p=0
\quad\text{at }y=0,1.
\]

The permeability is scalar and positive, with

\[
k(x,y)=\exp(\ell(x,y)),
\]

where the log-permeability field is parameterized by

\[
\ell(x,y)=\mu+a\,\Psi_\theta(x,y).
\]

Here \(\mu=\log k_{\mathrm{mean}}\), \(a=\log k_{\mathrm{amp}}\) in the implementation's parameter naming, and \(\theta\) rotates a prescribed spatial heterogeneity pattern. The angle rotates the scalar spatial pattern; it does not define an anisotropic permeability tensor.

The pattern is constructed in rotated coordinates

\[
x_r=\cos\theta\,x+\sin\theta\,y,\qquad
y_r=-\sin\theta\,x+\cos\theta\,y,
\]

using a fixed combination of trigonometric modes. The three investigated parameter intervals are

\[
\mu\in[-1,1],\qquad a\in[0.2,1.5],\qquad
\theta\in[0^\circ,180^\circ].
\]

These intervals define the computational study domain. They have not been established as physical operating limits for a particular geological formation.

### 2.2 Finite-volume discretization

The supplied solver uses a cell-centred finite-volume discretization on a 64 × 64 uniform grid for the main controlled experiments. Interior face permeabilities are harmonically averaged. Dirichlet boundary contributions use half-cell transmissibilities, while the top and bottom boundaries have zero normal flux. The resulting sparse linear system is solved using SciPy's sparse direct solver.

Separate solver checks reported during development included the constant-permeability analytical pressure solution, left/right flux balance, pressure invariance under a uniform shift of mean log-permeability, the corresponding flux scaling, and mesh-refinement checks. These checks support implementation consistency but do not, by themselves, establish all aspects of model validity or mesh-independent surrogate accuracy.

### 2.3 Structural parameter dependence

For this model, adding a constant \(c\) to \(\ell\) multiplies permeability by \(e^c\). Both the discrete operator and its Dirichlet forcing are scaled by the same factor, so the pressure solution is unchanged. The Darcy flux, however, scales with permeability. This gives an exact structural reason to exclude `logk_mean` from pressure regression while retaining it when calculating flux. The controlled regression experiment includes numerical checks of this behaviour.

The angle is treated as a bounded, non-periodic input over 0–180°. The endpoint fields were checked and are not identified as equivalent for the implemented heterogeneity pattern.

## 3. Investigation I: sampling and initial surrogate comparison

### 3.1 Design

The initial investigation compared four sampling strategies:

- uniform random sampling;
- Latin hypercube sampling (LHS);
- scrambled Sobol low-discrepancy sampling;
- uncertainty-driven adaptive sampling, initialized with LHS points and extended in batches using an integrated POD-GPR uncertainty criterion.

Training sizes were 16, 32, 64, 128, and 256, with three repetitions. A common validation set of 128 cases and a separate final test set of 256 cases were generated. The surrogate approaches were POD-GPR, which regresses POD coefficients, and a direct multi-output neural network mapping parameters to the full pressure field. Validation results were used to select the sampling strategy and training size for each model; the final selected models were refitted and evaluated on the held-out test set.

A Sobol global sensitivity analysis was then performed using the selected POD-GPR surrogate. The scalar quantity of interest (QoI) was the spatial mean of the pressure field. The analysis used independent uniform inputs over the stated bounds, a maximum base sample size of 4096, and 200 bootstrap resamples for confidence estimates.

### 3.2 Reported results

Validation selected the following configurations:

| Surrogate | Selected sampling | Training size | Validation mean relative L2 | Validation SD |
|---|---|---:|---:|---:|
| POD-GPR | LHS | 256 | 1.025979 × 10⁻³ | 7.021016 × 10⁻⁵ |
| Direct NN | Sobol | 256 | 9.768674 × 10⁻³ | 2.429706 × 10⁻³ |

On the held-out test set:

| Surrogate | Selected sampling | Global relative L2 | Mean case relative L2 | Maximum case relative L2 |
|---|---|---:|---:|---:|
| POD-GPR | LHS | 9.580484 × 10⁻⁴ | 8.601140 × 10⁻⁴ | 3.755522 × 10⁻³ |
| Direct NN | Sobol | 7.916819 × 10⁻³ | 6.856115 × 10⁻³ | 3.213562 × 10⁻² |

The Sobol indices for domain-averaged pressure were reported as:

| Parameter | First-order index \(S_1\) | Total-order index \(S_T\) |
|---|---:|---:|
| `logk_mean` | 0.000002 ± 0.000007 | approximately 0 |
| `logk_amp` | 0.069741 ± 0.020754 | 0.242958 ± 0.020037 |
| `pattern_angle_deg` | 0.757988 ± 0.043246 | 0.934503 ± 0.044640 |

### 3.3 Interpretation and limitation

For the selected pressure QoI and parameter bounds, orientation had the largest estimated influence. The amplitude's total-order effect was appreciably larger than its first-order effect, which is consistent with interactions involving this parameter. The mean log-permeability had negligible influence on the pressure QoI, consistent with the structural invariance described above.

The held-out POD-GPR error was lower than the direct-NN error in this particular selected comparison. However, the validation procedure selected LHS for POD-GPR and Sobol sampling for the direct NN. Because model family and sampling strategy differ simultaneously in the final comparison, the error difference cannot be interpreted as an isolated causal effect of regression method. The Sobol indices are also conditional on the chosen QoI, independent-uniform input assumption, parameter bounds, and surrogate. Their bootstrap intervals do not include surrogate-training variability or model-form uncertainty, and the indices concern mean pressure rather than flux.

This limitation motivated the controlled regression comparison below.

**Figure placeholder — sampling and validation results**

<!-- Insert the full-run validation error versus training size plot here. -->

`![Validation error versus training size](../figures/investigation1_validation_error_vs_samples.png)`

**Figure placeholder — pressure sensitivity**

<!-- Insert the full-run Sobol index plot here. -->

`![Sobol sensitivity indices for domain-averaged pressure](../figures/investigation1_sobol_indices.png)`

## 4. Investigation II: controlled POD regression comparison

### 4.1 Experimental controls

The second investigation was designed to compare regression methods without the sampling-strategy confounding present in the initial final comparison. The controlled study used:

- a 64 × 64 mesh;
- parameter bounds identical to those in Investigation I;
- pressure input features `logk_amp` and `pattern_angle_deg`;
- an independent LHS over these two pressure-active inputs, with `logk_mean` sampled independently for flux evaluation;
- training sizes 16, 32, 64, 128, and 256 during validation, with three repetitions;
- 128 validation cases and 256 held-out test cases;
- a common final training set of 256 cases and one common POD basis for the final head-to-head comparison.

For each training run, the POD mean and basis were fitted using training snapshots only. All regression candidates in a comparison used the same POD coefficients and basis. Candidate hyperparameters were selected using validation data; the final test set was not used for selection.

The six tested families were Gaussian process regression (GPR), an artificial neural network (ANN), polynomial regression with Ridge regularization, multiquadric radial-basis-function regression, tensor-product cubic B-spline regression, and thin-plate-spline RBF interpolation. The B-spline and thin-plate-spline methods are distinct approaches.

### 4.2 Held-out results at POD rank 32

The final shared-basis comparison produced the following results:

| Regression family | Pressure global relative L2 | Regression-only relative L2 | POD truncation relative L2 | Flux relative L2 |
|---|---:|---:|---:|---:|
| **GPR (RBF)** | **1.003983 × 10⁻³** | **1.442021 × 10⁻⁴** | **9.935730 × 10⁻⁴** | **1.167927 × 10⁻²** |
| Thin-plate spline | 1.651602 × 10⁻² | 1.648599 × 10⁻² | 9.935730 × 10⁻⁴ | 4.098775 × 10⁻² |
| Multiquadric RBF | 2.067018 × 10⁻² | 2.064614 × 10⁻² | 9.935730 × 10⁻⁴ | 9.248481 × 10⁻² |
| ANN | 2.224403 × 10⁻² | 2.222167 × 10⁻² | 9.935730 × 10⁻⁴ | 5.371581 × 10⁻² |
| Tensor cubic B-spline | 5.835189 × 10⁻² | 5.834302 × 10⁻² | 9.935730 × 10⁻⁴ | 2.128522 × 10⁻¹ |
| Polynomial (Ridge) | 7.042892 × 10⁻² | 7.042141 × 10⁻² | 9.935730 × 10⁻⁴ | 2.490470 × 10⁻¹ |

The validation-selected RBF-GPR candidate at `N=256` had a validation mean relative error of `9.749397 × 10⁻⁴` and standard deviation `2.187 × 10⁻⁵` across the reported repetitions. Its held-out global pressure error was `1.003983 × 10⁻³`, a close validation/test agreement for this split.

The decomposition identity was satisfied to numerical precision for all methods. For GPR, the POD truncation component was much larger than the regression-only component at rank 32. Pressure prediction error and flux prediction error were not interchangeable: GPR's pressure error was approximately 0.100%, while its flux error was approximately 1.168%. Flux depends on reconstructed pressure gradients and the permeability field, so its error can respond differently to pressure-field error.

The ANN reached its configured iteration limit (`2500`) and the run recorded a convergence warning. Its reported result is therefore not necessarily the performance of a fully converged ANN. GPR coefficient-space interval coverage was approximately 82.5% for nominal 90% intervals and 86.9% for nominal 95% intervals; these are coefficient-space diagnostics, not pointwise field-space coverage guarantees.

**Figure placeholder — held-out parity and spatial errors**

<!-- Insert full-run figures, not smoke-test figures. -->

`![Held-out pressure parity](../figures/investigation2_pressure_parity.png)`

`![Held-out pressure error field](../figures/investigation2_pressure_error_map.png)`

## 5. Investigation II continued: POD-rank sensitivity

### 5.1 Design

After the regression-family comparison, the RBF-GPR family was held fixed and the retained POD rank was varied over

\[
r\in\{8,16,24,32,48,64,96,128,192,255\}.
\]

The maximum numerical rank of the centred 256-snapshot training matrix was 255. The same training, validation, and test designs were reused at every rank (256, 128, and 256 cases, respectively). Each POD basis was fitted using training snapshots only. The RBF kernel family was fixed; GPR hyperparameters were fitted from the training coefficients at each rank. Neither validation nor test data were used to fit the POD basis or GP models. The held-out test set was not used to select a rank or kernel.

For each rank, total pressure error was separated into the orthogonal POD projection residual and the regression error within the retained POD subspace. In the global relative-L2 normalization used here, the squared component norms satisfy the corresponding Pythagorean decomposition up to numerical precision.

### 5.2 Held-out rank-sensitivity results

| POD rank | Captured energy | Pressure total relative L2 | POD truncation relative L2 | Regression relative L2 | Flux relative L2 |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.96664802 | 1.571829 × 10⁻² | 1.571782 × 10⁻² | 1.224349 × 10⁻⁴ | 1.089775 × 10⁻¹ |
| 16 | 0.99752438 | 4.377782 × 10⁻³ | 4.375537 × 10⁻³ | 1.401693 × 10⁻⁴ | 3.000003 × 10⁻² |
| 24 | 0.99964347 | 1.815754 × 10⁻³ | 1.810138 × 10⁻³ | 1.426986 × 10⁻⁴ | 1.676875 × 10⁻² |
| 32 | 0.99990574 | 1.003983 × 10⁻³ | 9.935730 × 10⁻⁴ | 1.442017 × 10⁻⁴ | 1.167927 × 10⁻² |
| 48 | 0.99999260 | 3.487902 × 10⁻⁴ | 3.166066 × 10⁻⁴ | 1.463380 × 10⁻⁴ | 3.942197 × 10⁻³ |
| 64 | 0.99999954 | 1.774019 × 10⁻⁴ | 9.564026 × 10⁻⁵ | 1.494134 × 10⁻⁴ | 1.086900 × 10⁻³ |
| 96 | 0.99999999 | 1.569470 × 10⁻⁴ | 2.600596 × 10⁻⁵ | 1.547774 × 10⁻⁴ | 7.794677 × 10⁻⁴ |
| 128 | approximately 1 | 1.565544 × 10⁻⁴ | 6.976765 × 10⁻⁶ | 1.563988 × 10⁻⁴ | 7.618677 × 10⁻⁴ |
| 192 | approximately 1 | 1.565990 × 10⁻⁴ | 1.409158 × 10⁻⁶ | 1.565927 × 10⁻⁴ | 7.646108 × 10⁻⁴ |
| 255 | approximately 1 | 1.566414 × 10⁻⁴ | 4.871766 × 10⁻⁷ | 1.566406 × 10⁻⁴ | 7.649625 × 10⁻⁴ |

The pressure error decreased rapidly as the POD rank increased from 8 to 64. Between ranks 64 and 96, the truncation error continued to fall while regression error remained near `1.5 × 10⁻⁴`. Beyond rank 96, the total pressure error changed very little because the regression component dominated. At rank 255, the truncation error was below `5 × 10⁻⁷`, while total pressure error remained approximately `1.566 × 10⁻⁴`.

The validation total pressure errors at ranks 96, 128, 192, and 255 were approximately `8.12 × 10⁻⁵`, `8.04 × 10⁻⁵`, `8.05 × 10⁻⁵`, and `8.06 × 10⁻⁵`, respectively. These values are very close. Rank 96 is therefore a useful practical knee for the current design; rank 128 has the smallest reported validation error, but the difference is small. The held-out test values are reported to characterize the curve, not to retrospectively select the rank.

Flux error also decreased substantially with increasing rank and reached approximately `7.6–7.8 × 10⁻⁴` from ranks 96 to 255. The rank with the smallest observed flux error was 128, but the differences among ranks 96, 128, 192, and 255 are small relative to the overall improvement from low ranks.

**Figure placeholder — pressure error decomposition versus rank**

`![Pressure error decomposition versus POD rank](../figures/pod_rank_pressure_error_decomposition.png)`

**Figure placeholder — flux error versus rank**

`![Flux error versus POD rank](../figures/pod_rank_flux_error.png)`

### 5.3 Interpretation

The rank-sensitivity results identify a transition between two error regimes:

- At low rank, POD truncation dominates, so adding modes substantially improves the pressure prediction.
- At higher rank, regression error becomes comparable to and then larger than truncation error. Adding further modes improves the representational capacity of POD but does not materially improve the learned surrogate.

This is why captured POD energy alone is not a sufficient criterion for choosing the reduced dimension. The practically useful rank depends on both the projection error and the ability of the regression model to learn the corresponding coefficients from the available training data.

The result does not prove that rank 96 is universally optimal, nor that GPR remains the best regression family at every rank. The study uses one fixed training design and one regression family for the rank sweep. In addition, the number of coefficient models grows with rank while the training set remains fixed at 256 samples. The regression floor may therefore be related to the finite training sample size, the chosen kernel/model, or both.

## 6. Integrated discussion

The two investigations form a connected sequence. The initial study explored sampling and compared two surrogate approaches, but its selected final configurations differed in sampling strategy. The controlled comparison then isolated regression-family differences under a shared POD basis and common final data design. Finally, the rank sweep showed that the error of the selected GPR surrogate was not determined by regression alone: at rank 32, POD truncation was the dominant pressure-error contribution, while at ranks above roughly 96 the regression component became the limiting factor.

Together, the results support three study-specific observations:

1. **Sampling and model configuration matter together.** The initial study identified promising configurations, but its final POD-GPR versus NN comparison was not a pure algorithm comparison because the selected sampling strategies differed.
2. **Regression family matters under controlled conditions.** At rank 32 and the common final design, RBF-GPR achieved the lowest held-out pressure and flux errors among the six methods tested.
3. **Reduced-basis dimension creates an error trade-off.** Increasing POD rank reduces projection error, but the overall surrogate error eventually reaches a regression-limited floor.

The results also reinforce the need to assess pressure and flux separately. Pressure is invariant to a global permeability multiplier in this boundary-value problem, whereas flux is not. Moreover, flux depends on pressure gradients, so a small pressure-field error does not automatically imply an equally small flux error.

## 7. Limitations and next steps

The following limitations should accompany any presentation of the results:

- **Parameter domain:** the parameter bounds are exploratory and may not represent a particular geological system.
- **Model scope:** the problem is steady, scalar, isotropic Darcy flow with a prescribed synthetic heterogeneity pattern and fixed boundary conditions.
- **Spatial discretization:** the main experiments use a 64 × 64 mesh. Separate solver checks were performed, but the surrogate study itself does not establish mesh independence.
- **Initial comparison:** the final POD-GPR and direct-NN configurations use different validation-selected sampling strategies; their test errors do not isolate regression method.
- **Controlled comparison:** the final six-family comparison uses one common final training set and one held-out test set. It does not provide a multi-split estimate of test uncertainty.
- **ANN convergence:** the ANN reached the iteration limit and recorded a convergence warning.
- **GPR uncertainty:** nominal interval coverage is reported in POD-coefficient space and appears below nominal coverage; this is not a field-space uncertainty calibration result.
- **Rank sensitivity:** one fixed training/validation/test design is used. The observed rank knee is conditional on this design and the fixed RBF-GPR family.
- **Regression floor:** the rank sweep does not determine whether the high-rank regression floor is caused mainly by the 256-sample training budget, the regression model, or both.
- **Sensitivity analysis:** Sobol indices apply to domain-averaged pressure under independent uniform inputs over the stated bounds; bootstrap intervals do not include surrogate-training or model-form uncertainty.

A focused next experiment, if further work is warranted, would vary training sample size at fixed ranks near 96 and 128, using multiple independent training designs and a fixed test set. This would test whether the high-rank regression floor decreases with additional training data. It should be undertaken only if it addresses a specific research question or is relevant to a prospective research group.

## 8. Conclusion

This independent computational investigation developed and assessed a reduced-order surrogate workflow for a parameterized steady Darcy-flow problem. The initial sampling study identified promising POD-GPR and direct-NN configurations and provided a global sensitivity analysis for mean pressure. A subsequent controlled comparison showed that RBF-GPR achieved the lowest held-out pressure and flux errors among six tested regression families at POD rank 32. The POD-rank sensitivity study then demonstrated that increasing rank substantially reduced pressure error until regression error became the dominant limitation at approximately rank 96 for the current 256-sample design.

The central conclusion is not that one regression method or POD rank is universally best. Rather, the useful accuracy of a POD-based surrogate depends on the interaction between sampling, reduced-basis truncation, regression capability, and the quantity being predicted. The study provides a reproducible starting point for further work on reduced-order modelling and data-driven surrogate methods for flow problems.

## Reproducibility note

The repository includes the solver and scripts for the three computational components. The rank-sensitivity summary and CSV are included. The original full raw-output bundle for the initial sampling investigation was not present in the materials assembled for this report; its reported headline metrics are preserved here, but the corresponding full CSV/NPZ artifacts should be added before claiming complete reproduction from the repository alone.
