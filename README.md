# Reduced-order surrogates for parameterized Darcy flow

A small side project: how well can a fast surrogate model replace a Darcy-flow solver, and what limits its accuracy? I looked at three things: how the training simulations are chosen, which regression model is used, and how many POD modes are kept.

I was inspired by the work of **Prof. Dr. Denise Degen** and the Platypus paper (Hayek et al., 2026, see [References](#references)), and wanted to try related questions hands-on on a much smaller problem. This is a personal exploration, not a paper. I am a geologist (my PhD work included reservoir modelling and machine learning), AI tools wrote most of the code and drafted the text. I chose the questions, directed the work, ran the experiments and checked the results, and I take responsibility for what is published here. Every result is only as good as the setup described below. I added a Bayesian part at the end because Bayesian methods were part of my PhD work and I wanted to see how they fit this problem. Corrections and feedback are very welcome.

## In short

1. **POD + Gaussian-process regression (POD-GPR) beat a direct neural network by about 8×** on held-out test cases (relative L2 error 9.6e-4 vs 7.9e-3), with 256 training solves each. The two used different sampling strategies, so this is not a clean comparison.
2. **With the data and POD basis held fixed, GPR beat five other regressors** (pressure error 1.0e-3 vs 1.7e-2 to 7.0e-2).
3. **More POD modes help only up to a point.** Pressure error falls until about 64 to 96 modes, then flattens at about 1.6e-4, where the regression, not the POD truncation, is the limit.
4. **The pattern angle controls most of the variance in mean pressure** (Sobol first-order 0.76, total 0.93). The mean log-permeability has no effect on pressure, as expected.
5. **Does surrogate error distort Bayesian inference?** Only when the measurement noise is smaller than the surrogate's own error. Then the posterior is overconfident, and modelling the surrogate error fixes most of it (section 5).

## The problem

Steady Darcy flow on the unit square, -∇·(k ∇p) = 0, with p = 1 on the left, p = 0 on the right and no flow through the top and bottom. Permeability is k = exp(μ + a·Ψ_θ(x, y)), where Ψ_θ is a fixed synthetic heterogeneity pattern rotated by an angle θ.

| Parameter | Range | Note |
|---|---|---|
| `logk_mean` (μ) | -1 to 1 | Scales k by a constant. Leaves pressure unchanged, only scales flux. |
| `logk_amp` (a) | 0.2 to 1.5 | Strength of the heterogeneity pattern. |
| `pattern_angle_deg` (θ) | 0° to 180° | Rotates the pattern. Called `logk_angle` in the sampling-study plots. |

The pipeline is: full-order solver → parameter sampling → snapshot matrix → POD basis → GPR for the POD coefficients → reconstructed pressure field → error evaluation.

The solver is a cell-centred finite-volume code on a 64 × 64 grid (`src/darcy.py`). The ranges are study choices, not real geological limits. Because `logk_mean` cannot change the pressure, the pressure surrogates in the controlled study (sections 3 and 4) use only amplitude and angle. "Flux" throughout means the total inflow through the left boundary. It is computed from the predicted pressure in the first column of cells and uses all three parameters.

## Results

### 1. Sampling strategy, and POD-GPR vs a direct neural network

I compared four ways of choosing training simulations: random, Latin hypercube (LHS), scrambled Sobol sequence, and adaptive (start from LHS points, then add batches where the POD-GPR is most uncertain). Each was used with POD-GPR and with a direct neural network (NN) that maps parameters straight to the pressure field. Training sizes were 16 to 256, with 3 repeats each. The configurations chosen on validation data were refitted and tested on 256 held-out cases. Figure 1 shows the validation error.

![Figure 1](figures/validation_error_vs_samples.png)

**Figure 1.** Validation error (mean ± SD of 3 runs) against the number of full-order training solves, for each sampling strategy. Left: POD-GPR. Right: direct NN.

- **POD-GPR needs far fewer simulations.** At 128 solves it is at 3e-3 to 1.2e-2, while the NN is still at 4e-2 to 6e-2.
- **At 256 solves, random, LHS and adaptive sampling are indistinguishable** for POD-GPR (all about 1.0 to 1.1e-3). Sobol sampling was erratic (individual runs: 8.7e-4, 5.5e-3 and 1.8e-2).
- **Adaptive sampling had the lowest mean error at 32, 64 and 128 solves**, but with 3 repeats and overlapping spread this is a hint, not a result.

The final models were then tested on 256 held-out cases (table below and Figure 2).

| Model | Sampling (picked on validation) | Training solves | POD modes | Test rel. L2 (global) | Worst test case |
|---|---|---|---|---|---|
| POD-GPR | LHS | 256 | 32 | 9.58e-4 | 3.76e-3 |
| Direct NN | Sobol | 256 | - | 7.92e-3 | 3.21e-2 |

![Figure 2](figures/sampling_final_test_parity.png)

**Figure 2.** Surrogate vs full-order pressure on the 256 test cases. Left: POD-GPR. Right: direct NN.

For POD-GPR the largest pointwise error on any test case is about 0.008 on a pressure range of 0 to 1. Figure 3 shows that worst case: the error is made of smooth patches of alternating sign, not random noise.

![Figure 3](figures/worst_case_pressure_error.png)

**Figure 3.** POD-GPR pressure error (surrogate minus full-order) for the worst test case. Positive and negative patches alternate; the largest magnitude is about 0.008.

**Two caveats.** The final models used different sampling strategies (LHS vs Sobol), so the 8× gap does not isolate the regression method. That is why section 3 exists. Also, POD-GPR used about 32 modes here (the number picked automatically by keeping 99.99% of the snapshot energy), and section 4 suggests that at 32 modes the error is mostly POD truncation. So 9.6e-4 is probably not the best POD-GPR can do with this data. I have not re-run this study with more modes.

### 2. Which parameters matter for mean pressure?

Sobol indices of the domain-averaged pressure, computed on the final POD-GPR surrogate with independent uniform inputs over the ranges above (4096 base samples, 200 bootstrap resamples).


| Parameter | First-order | Total-order |
|---|---|---|
| `logk_mean` | ≈ 0 | ≈ 0 |
| `logk_amp` | 0.070 ± 0.021 | 0.243 ± 0.020 |
| `pattern_angle_deg` | 0.758 ± 0.043 | 0.935 ± 0.045 |

![Figure 4](figures/sobol_indices.png)

**Figure 4.** First- and total-order Sobol indices of mean pressure with 95% bootstrap intervals (4096 base samples). In the plot the angle is labelled `logk_angle`.

Figure 4 shows the same numbers. The angle dominates. For the amplitude, the total-order index is much larger than the first-order one, which points to interaction with the angle. The zero for `logk_mean` is a sanity check, not a finding. The intervals reflect Sobol sampling only, not surrogate error, and the analysis covers mean pressure, not flux.

### 3. Six regression families on a shared POD basis

Section 1 changed the model and the sampling at the same time, so here I held everything else fixed: the same 256 training cases, the same POD basis (32 modes, again from the 99.99% energy rule), the same test set, and hyperparameters chosen on validation data only.

| Regression | Pressure rel. L2 | Flux rel. L2 |
|---|---|---|
| **GPR (RBF)** | **1.0e-3** | **1.17e-2** |
| Thin-plate spline | 1.65e-2 | 4.10e-2 |
| Multiquadric RBF | 2.07e-2 | 9.25e-2 |
| ANN* | 2.22e-2 | 5.37e-2 |
| Tensor cubic B-spline | 5.84e-2 | 2.13e-1 |
| Polynomial (Ridge) | 7.04e-2 | 2.49e-1 |

Figure 5 shows predicted against true pressure for each regressor, and Figure 6 where in the domain the errors occur.

![Figure 5](figures/regression_final_test_parity.png)

**Figure 5.** Surrogate vs full-order pressure on the test set for each of the six regressors.

![Figure 6](figures/final_spatial_error_maps.png)

**Figure 6.** Mean absolute pressure error over the test cases, for each regressor. Each panel has its own colour scale, so compare the colour-bar values, not the colours: the GPR maximum is below 0.001, the polynomial and B-spline maxima are about 0.05. For the other regressors the errors are largest in the upper part of the domain.

At 32 modes the GPR error is mostly POD truncation (9.9e-4), not regression (1.4e-4). For the other five families nearly all of the error is regression error. \* The ANN stopped at its iteration limit (2500) without converging, so it may improve with longer training and its number is not a fully converged result.

### 4. How many POD modes?

With the regression fixed to RBF-GPR and the same data at every step, I varied the number of retained POD modes and split the test error into POD truncation (what the basis cannot represent) and regression error (what the GPR gets wrong inside the basis).

| POD modes | Pressure error | Truncation part | Regression part | Flux error | GPR fit time |
|---|---|---|---|---|---|
| 8 | 1.57e-2 | 1.57e-2 | 1.2e-4 | 1.09e-1 | 6 s |
| 32 | 1.00e-3 | 9.9e-4 | 1.4e-4 | 1.17e-2 | 24 s |
| 64 | 1.77e-4 | 9.6e-5 | 1.5e-4 | 1.09e-3 | 58 s |
| 96 | 1.57e-4 | 2.6e-5 | 1.5e-4 | 7.8e-4 | 73 s |
| 128 | 1.57e-4 | 7.0e-6 | 1.6e-4 | 7.6e-4 | 85 s |
| 255 | 1.57e-4 | 4.9e-7 | 1.6e-4 | 7.6e-4 | 154 s |

![Figure 7](figures/pressure_error_vs_pod_rank.png)

**Figure 7.** Held-out pressure error against POD rank, split into POD truncation and regression error.

![Figure 8](figures/flux_error_vs_pod_rank.png)

**Figure 8.** Held-out flux error against POD rank.

In Figure 7, the error falls quickly up to about 64 modes, while truncation dominates. From about 96 modes on, the total is flat at about 1.6e-4 because regression error has become the limit, and extra modes only add cost: fit time roughly doubles from 96 to 255 modes, and prediction time per case grows about 5×. The validation errors for 96 to 255 modes are nearly identical (8.0 to 8.1e-5), so I would call 96 a practical knee **for this 256-sample design**, not a general optimum.

Flux error (Figure 8) flattens too, at about 7.6e-4, but it is 5 to 12× larger than the pressure error at every rank. That gap is the starting point of idea 3 below.

### 5. Using the surrogate for Bayesian inference

The earlier sections measure surrogate error as a field error. Here I ask a practical question: if you use the surrogate to infer parameters from noisy data, does its error bias or over-sharpen the answer? I chose this because Bayesian inference is where I have the most experience, and it turns the rank findings into a downstream test.

The numbers below are from one full run of `src/bayesian_inversion.py` (random seed 20261112).
**Setup**
- Unknowns: `logk_amp` and `pattern_angle_deg`, uniform prior over the ranges above. `logk_mean` is left out because pressure does not depend on it.
- Data: pressure at 36 sensors from the **full solver** plus Gaussian noise of known size, so the true parameters are known.
- Posteriors by MCMC, for POD ranks 8, 32 and 64 (same GPR, 256 training solves):
  - **full solver**: the exact reference posterior
  - **surrogate, naive**: the surrogate is treated as exact
  - **surrogate, with discrepancy**: the surrogate's error at the 36 sensors is modelled as a Gaussian with a mean and a full covariance matrix across the sensors (Ledoit-Wolf shrinkage), estimated from 128 validation cases. These cases are separate from the training set and from the calibration truths
- Checks: three case studies compared with the full-solver posterior, and a calibration test over 100 truths drawn from the prior (how often the truth lies inside the 90% posterior region).

Figure 9 shows example posteriors and Figure 10 the calibration test.

![Figure 9](figures/fig2_case_posteriors.png)

**Figure 9.** Posterior regions (50% and 90%) for three true parameter sets and three noise levels, POD rank 32.

![Figure 10](figures/fig5_calibration_coverage.png)

**Figure 10.** Share of 100 truths inside the 90% posterior region, by POD rank and noise level. Target is 0.90.

**What I saw**
- **The surrogate matters once noise is below its error.** At the sensors the rank-32 surrogate has an RMS error of about 5e-4. At noise 3e-3 and above, naive and full-solver posteriors are close (naive coverage 88 to 89%). Around 1e-3 the naive one starts to drift (82%).
- **Rank 32, noise 1e-4: the naive posterior is overconfident.** The truth was inside its 90% region for only 17% of truths, and posterior means were shifted by several reference standard deviations. With the discrepancy model coverage was 92%.
- **Rank 8 is overconfident for all but the largest noise.** Naive coverage was 1% to 24% at noise 1e-4 to 1e-3. The discrepancy version is then too cautious (98 to 100% coverage at noise 3e-3 and below).
- **Rank 64: the surrogate error is below the noise,** so both versions agree.
- **The price of the discrepancy model is width.** At the smallest noise its posteriors are about 3× wider than the full-solver one. That is honest about the surrogate, but it also shows the limit of what this surrogate can resolve.

## What this does not show

- The ranges and the synthetic heterogeneity pattern are not tied to a real formation. The problem is steady, scalar and isotropic.
- Everything uses one 64 × 64 grid, so mesh independence is not tested. The study measures how well the surrogate reproduces this discretized solver, not the accuracy of the physical solution.
- Sections 3 and 4 each use a single train/test split. The error floor in section 4 is about 8e-5 on the validation set but 1.6e-4 on the test set, so its exact value depends on the data design.
- The Sobol indices cover mean pressure only and carry no surrogate-training uncertainty.
- GPR uncertainty intervals were checked in POD-coefficient space only, and they under-covered (about 82.5% for a nominal 90% and 86.9% for a nominal 95%).
- The Bayesian test uses synthetic data from the same solver, known i.i.d. Gaussian noise and one fixed training set. It says nothing about real measurements or model misfit. With 100 truths per setting, coverage values are only good to about ±0.06.
- The discrepancy model uses one mean and one covariance for the whole parameter box, although surrogate error varies with the parameters, and whether the residuals are really Gaussian was not tested. The reference chains are short (R-hat at most 1.03, effective sample size at least about 220 in my run).
- Summary tables of the sampling study are included, but the raw data archive from that run was not preserved, so a re-run may not reproduce those exact numbers.

## Ideas for future work

1. **Training size vs POD rank.** Is the ~1.6e-4 floor a data limit? Try 128 to 768 training samples at 96 and 128 modes with RBF-GPR, and track truncation error, regression error, flux error and cost. *This is the next experiment I would run.*
2. **Regression families at higher rank.** Does GPR stay best at 96 modes? Re-running the sampling comparison with more modes would also show how much of section 1 was truncation.
3. **Flux.** Why is flux error 5 to 12× the pressure error? Part of it may be an artefact of how flux is computed: the left-boundary inflow is a finite difference across half a grid cell next to the boundary, where the pressure drop is small, so a tiny pressure error there gets magnified. Compare it with flux through interior faces, a direct flux surrogate, or a separate POD basis for flux.
4. **Adaptive sampling.** Does uncertainty-driven sampling reach the same accuracy with fewer solves at equal budget? My 3-repeat sweep hints at it but is too noisy to say.
5. **Mesh refinement and generalization.** Repeat the study at 32 × 32 and 128 × 128 to see whether the conclusions depend on resolution, and train at 32 × 32 or 64 × 64 and test at 128 × 128.
6. **Better Bayesian treatment.** Let the surrogate-error model vary with the parameters (for example a GP on the residuals), use the GPR's own predictive variance, repeat over several training designs, and try a harder inverse problem with more than two unknowns or less informative data.
7. **Bayesian sensitivity analysis.** Propagate the GPR posterior into the Sobol indices to get intervals that include surrogate uncertainty, which section 2 lacks.
8. **Trust-aware surrogates**. Can a surrogate say not only what it predicts, but also when its prediction should not be trusted? The GPR intervals here under-covered (see “What this does not show”), so a first step would be a calibrated error estimate, for example conformal prediction or a reduced-basis error estimator. It would be tested on held-out cases for coverage and for how often it wrongly accepts or rejects a prediction. A second step is to use that estimate to decide where extra full-order simulations would help most, and to compare this with simply adding LHS points at equal budget. A third step is to expose the surrogate and its trust flag as a tool that an AI agent can call, falling back to the full solver when the flag says the prediction is unreliable.

If any of these overlap with your work, I would be happy to talk.

## Repository

```
src/
  darcy.py                         finite-volume solver
  sampling_surrogate_study.py      sampling study, POD-GPR vs NN, Sobol analysis
  experiment_a.py                  six-regressor comparison
  pod_rank_sensitivity.py          POD-rank sweep
  bayesian_inversion.py            Bayesian inference with the surrogate
  smoke_test.py                    quick workflow check (not research results)
docs/
  technical_report.md              longer report with all tables
  limitations_and_reproducibility.md
results/
  initial_surrogate_investigation/
  regression_comparison/
  pod_rank_sensitivity/
  bayesian_inversion/
figures/
```

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH=src                # PowerShell: $env:PYTHONPATH = "src"
python src/smoke_test.py             # optional: reduced run of the regression comparison
python src/sampling_surrogate_study.py   # writes corrected_surrogate_results/
python src/experiment_a.py               # writes experiment_a_results/
python src/pod_rank_sensitivity.py       # writes pod_rank_sensitivity_results/
python src/bayesian_inversion.py         # writes bayesian_inversion_results/ (about 20 min on 2 CPUs)
python src/bayesian_inversion.py --quick # short check of the workflow
```

The sampling study needs `SALib` for the Sobol analysis. Smoke-test runs use coarse meshes and few samples. They are a software check only, and none of the numbers above come from them.

## Related work

Platypus (Hayek et al., 2026) builds surrogates with the non-intrusive reduced-basis method: POD for the basis, then Gaussian-process regression or a neural network for the coefficients, with a POD energy threshold, Latin hypercube sampling and Sobol analysis via SALib. This repository is a small, Darcy-flow version of the same recipe, written independently with a finite-volume solver, and it is not a reproduction of their package. The Platypus outlook mentions adaptive sampling and low-discrepancy sequences as possible extensions, which section 1 tries on the toy problem.

## References

Hayek, J. N., Cacace, M., Faber, M., Gruzdeva, Y., and Degen, D.: Platypus: a Python package for automated surrogate-assisted analysis of process-based simulations via the non-intrusive reduced-basis method, Adv. Geosci., 68, 1–11, https://doi.org/10.5194/adgeo-68-1-2026, 2026.

## Contact

Hasan, Berlin · hasan.alatrash87@gmail.com
