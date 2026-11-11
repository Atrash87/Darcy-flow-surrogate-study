# Reduced-order surrogates for parameterized Darcy flow

A small side project: how well can a fast surrogate model replace a Darcy-flow solver, and what limits its accuracy? I looked at three things: how the training simulations are chosen, which regression model is used, and how many POD modes are kept.

I was inspired by the work of **[Dr. NAME, paper/topic]** and wanted to try related questions hands-on. This is a personal exploration, not a paper. I am a geologist with a data-analytics background, I used AI tools to help write the code and the text, and every result here is only as good as the setup described below. Corrections and feedback are very welcome.

## In short

1. **POD + Gaussian-process regression (POD-GPR) beat a direct neural network by about 8×** on held-out test cases (relative L2 error 9.6e-4 vs 7.9e-3). The two used different sampling strategies, so this is not a clean comparison.
2. **With the data and POD basis held fixed, GPR beat five other regressors** (pressure error 1.0e-3 vs 1.7e-2 to 7.0e-2).
3. **More POD modes help only up to a point.** Pressure error falls until about 64 to 96 modes, then flattens at about 1.6e-4, where the regression, not the POD truncation, is the limit.
4. **The pattern angle controls most of the variance in mean pressure** (Sobol first-order 0.76, total 0.93). The mean log-permeability has no effect on pressure, as expected.

## The problem

Steady Darcy flow on the unit square, -∇·(k ∇p) = 0, with p = 1 on the left, p = 0 on the right and no flow through the top and bottom. Permeability is k = exp(μ + a·Ψ_θ(x, y)), where Ψ_θ is a fixed synthetic heterogeneity pattern rotated by an angle θ.

| Parameter | Range | Note |
|---|---|---|
| `logk_mean` (μ) | -1 to 1 | Scales k by a constant. Leaves pressure unchanged, only scales flux. |
| `logk_amp` (a) | 0.2 to 1.5 | Strength of the heterogeneity pattern. |
| `pattern_angle_deg` (θ) | 0° to 180° | Rotates the pattern. Called `logk_angle` in the sampling-study plots. |

The solver is a cell-centred finite-volume code on a 64 × 64 grid (`src/darcy.py`). The ranges are study choices, not real geological limits. Because `logk_mean` cannot change the pressure, the pressure surrogates in the controlled study (sections 3 and 4) use only amplitude and angle. "Flux" throughout means the total inflow through the left boundary. It is computed from the predicted pressure in the first column of cells and uses all three parameters.

## Results

### 1. Sampling strategy, and POD-GPR vs a direct neural network

I compared four ways of choosing training simulations: random, Latin hypercube (LHS), scrambled Sobol sequence, and adaptive (start from LHS points, then add batches where the POD-GPR is most uncertain). Each was used with POD-GPR and with a direct neural network (NN) that maps parameters straight to the pressure field. Training sizes were 16 to 256, with 3 repeats each. The configurations chosen on validation data were refitted and tested on 256 held-out cases.

> 🖼️ **FIGURE 1 — insert here:** `figures/investigation1_validation_error_vs_samples.png`

*Validation error (mean ± SD of 3 runs) against the number of full-order training solves, for each sampling strategy. Left: POD-GPR. Right: direct NN.*

- **POD-GPR needs far fewer simulations.** At 128 solves it is at 3e-3 to 1.2e-2, while the NN is still at 4e-2 to 6e-2.
- **At 256 solves, random, LHS and adaptive sampling are indistinguishable** for POD-GPR (all about 1.0 to 1.1e-3). Sobol sampling was erratic (individual runs: 8.7e-4, 5.5e-3 and 1.8e-2).
- **Adaptive sampling had the lowest mean error at 32, 64 and 128 solves**, but with 3 repeats and overlapping spread this is a hint, not a result.

| Model | Sampling (picked on validation) | Training solves | POD modes | Test rel. L2 (global) | Worst test case |
|---|---|---|---|---|---|
| POD-GPR | LHS | 256 | 32 | 9.58e-4 | 3.76e-3 |
| Direct NN | Sobol | 256 | - | 7.92e-3 | 3.21e-2 |

> 🖼️ **FIGURE 2 — insert here:** `figures/investigation1_parity_podgpr_vs_nn.png`

*Surrogate vs full-order pressure on the 256 test cases. Left: POD-GPR. Right: direct NN.*

For POD-GPR the largest pointwise error on any test case is about 0.008 on a pressure range of 0 to 1, and it is spatially structured rather than random.

**Two caveats.** The final models used different sampling strategies (LHS vs Sobol), so the 8× gap does not isolate the regression method. That is why section 3 exists. Also, POD-GPR used about 32 modes here (the number picked automatically by keeping 99.99% of the snapshot energy), and section 4 suggests that at 32 modes the error is mostly POD truncation. So 9.6e-4 is probably not the best POD-GPR can do with this data. I have not re-run this study with more modes.

### 2. Which parameters matter for mean pressure?

Sobol indices of the domain-averaged pressure, computed on the final POD-GPR surrogate with independent uniform inputs over the ranges above (4096 base samples, 200 bootstrap resamples). The estimates were stable from about 2048 base samples. <!-- TODO-VERIFY: read off the convergence plot; no convergence data file was available -->


| Parameter | First-order | Total-order |
|---|---|---|
| `logk_mean` | ≈ 0 | ≈ 0 |
| `logk_amp` | 0.070 ± 0.021 | 0.243 ± 0.020 |
| `pattern_angle_deg` | 0.758 ± 0.043 | 0.935 ± 0.045 |

> 🖼️ **FIGURE 3 — insert here:** `figures/investigation1_sobol_indices.png`

*First- and total-order Sobol indices with 95% bootstrap intervals (4096 base samples). In the plot the angle is labelled `logk_angle`.*

The angle dominates. For the amplitude, the total-order index is much larger than the first-order one, which points to interaction with the angle. The zero for `logk_mean` is a sanity check, not a finding. The intervals reflect Sobol sampling only, not surrogate error, and the analysis covers mean pressure, not flux.

### 3. Six regression families on a shared POD basis

Section 1 changed the model and the sampling at the same time, so here I held everything else fixed: the same 256 training cases, the same POD basis (32 modes, again from the 99.99% energy rule), the same test set, and hyperparameters chosen on validation data only.

| Regression | Pressure rel. L2 | Flux rel. L2 |
|---|---|---|
| **GPR (RBF)** | **1.0e-3** | **1.17e-2** |
| Thin-plate spline | 1.65e-2 | 4.10e-2 |
| Multiquadric RBF | 2.07e-2 | 9.25e-2 |
| ANN | 2.22e-2 | 5.37e-2 |
| Tensor cubic B-spline | 5.84e-2 | 2.13e-1 |
| Polynomial (Ridge) | 7.04e-2 | 2.49e-1 |

> 🖼️ **FIGURE 4 — insert here (regenerate from the full N = 256 run):** `figures/investigation2_pressure_parity.png`

*Surrogate vs full-order pressure on the test set for each of the six regressors.*

<!-- TODO-VERIFY: only the GPR row is confirmed (it matches the rank-32 row of the rank sweep). The other five rows, the ANN iteration limit and the truncation/regression split come from technical_report.md. Check them against experiment_a_results/final_heldout_test_results.csv from the full run. -->
At 32 modes the GPR error is mostly POD truncation (9.9e-4), not regression (1.4e-4). For the other five families nearly all of the error is regression error. The ANN stopped at its iteration limit (2500) without converging, so its number is not a fully converged result.

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

> 🖼️ **FIGURE 5 — insert here:** `figures/pod_rank_pressure_error_decomposition.png`

*Held-out pressure error against POD rank, split into POD truncation and regression error.*

> 🖼️ **FIGURE 6 — insert here:** `figures/pod_rank_flux_error.png`

*Held-out flux error against POD rank.*

Error falls quickly up to about 64 modes, while truncation dominates. From about 96 modes on, the total is flat at about 1.6e-4 because regression error has become the limit, and extra modes only add cost: fit time roughly doubles from 96 to 255 modes, and prediction time per case grows about 5×. The validation errors for 96 to 255 modes are nearly identical (8.0 to 8.1e-5), so I would call 96 a practical knee **for this 256-sample design**, not a general optimum.

Flux error flattens too, at about 7.6e-4, but it is 5 to 12× larger than the pressure error at every rank. That gap is the starting point of idea 3 below.

## What this does not show

- The ranges and the synthetic heterogeneity pattern are not tied to a real formation. The problem is steady, scalar and isotropic.
- Everything uses one 64 × 64 grid, so mesh independence is not tested.
- Sections 3 and 4 each use a single train/test split. The error floor in section 4 is about 8e-5 on the validation set but 1.6e-4 on the test set, so its exact value depends on the data design.
- The Sobol indices cover mean pressure only and carry no surrogate-training uncertainty.
- GPR uncertainty intervals were checked in POD-coefficient space only, and they under-covered (about 82.5% for a nominal 90%). <!-- TODO-VERIFY: 82.5% comes from technical_report.md; check coefficient_coverage90 in the full-run final_heldout_test_results.csv -->
- Summary tables of the sampling study are included, but the raw data archive from that run was not preserved, so a re-run may not reproduce those exact numbers.

## Ideas for future work

1. **Training size vs POD rank.** Is the ~1.6e-4 floor a data limit? Try 128 to 768 training samples at 96 and 128 modes with RBF-GPR, and track truncation error, regression error, flux error and cost. *This is the next experiment I would run.*
2. **Regression families at higher rank.** Does GPR stay best at 96 modes? Re-running the sampling comparison with more modes would also show how much of section 1 was truncation.
3. **Flux.** Why is flux error 5 to 12× the pressure error? Part of it may be an artefact of how flux is computed: the left-boundary inflow is a finite difference across half a grid cell next to the boundary, where the pressure drop is small, so a tiny pressure error there gets magnified. Compare it with flux through interior faces, a direct flux surrogate, or a separate POD basis for flux.
4. **Adaptive sampling.** Does uncertainty-driven sampling reach the same accuracy with fewer solves at equal budget? My 3-repeat sweep hints at it but is too noisy to say.
5. **Mesh generalization.** Train at 32 × 32 or 64 × 64 and test at 128 × 128.

If any of these overlap with your work, I would be happy to talk.

## Repository

```
src/
  darcy.py                         finite-volume solver
  sampling_surrogate_study.py      sampling study, POD-GPR vs NN, Sobol analysis
  experiment_a.py                  six-regressor comparison
  pod_rank_sensitivity.py          POD-rank sweep
  smoke_test.py                    quick workflow check (not research results)
docs/
  technical_report.md              longer report with all tables
  limitations_and_reproducibility.md
results/
  initial_surrogate_investigation/
  regression_comparison/
  pod_rank_sensitivity/
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
```

The sampling study needs `SALib` for the Sobol analysis. Smoke-test runs use coarse meshes and few samples. They are a software check only, and none of the numbers above come from them.

## Contact

Hasan, Berlin · [email / LinkedIn / YouTube]
