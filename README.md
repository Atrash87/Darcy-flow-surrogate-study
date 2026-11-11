# Reduced-order surrogate modelling for parameterized steady Darcy flow

An independent computational study of sampling strategies, surrogate regression, and proper orthogonal decomposition (POD) rank for a parameterized steady Darcy-flow problem.

> **Scope.** This repository documents a computational investigation and its current results. It is not a reproduction of another group's complete study, a validated field-scale model, or a claim that one surrogate method is universally superior. Results are conditional on the model, discretization, parameter bounds, sampling designs, and evaluation protocol described below.

## Research question

How do sampling design, surrogate regression, and POD representation affect the accuracy and computational cost of predicting pressure fields and boundary flux in a parameterized Darcy-flow model?

The work is organized as **two connected investigations**:

1. **Sampling and initial surrogate comparison** — compare sampling designs and an initial POD-GPR surrogate with a direct neural-network baseline; examine global sensitivity of a pressure quantity of interest.
2. **Controlled POD-regression study** — compare regression families on a shared POD representation, then vary POD rank to separate representation error from regression error.

The second investigation follows from a limitation identified in the first: the initial final comparison selected different sampling strategies for the two surrogate approaches, so it could not isolate the effect of regression method alone.

## Problem

The steady scalar Darcy equation is solved on the unit square:

\[
-\nabla\cdot(k(x,y)\nabla p)=0,
\]

with fixed pressure on the left and right boundaries and no-flow conditions on the top and bottom. Permeability is positive and spatially heterogeneous, parameterized by

\[
(\log k_{\mathrm{mean}},\; a,\; \theta),
\]

where the angle rotates a scalar spatial heterogeneity pattern. The investigated parameter bounds are

| Parameter | Lower | Upper |
|---|---:|---:|
| `logk_mean` | -1.0 | 1.0 |
| `logk_amp` | 0.2 | 1.5 |
| `pattern_angle_deg` | 0° | 180° |

These are the bounds used in the experiments; they should be treated as study-domain choices, not independently established physical limits. The main surrogate experiments use a 64 × 64 cell-centred finite-volume discretization.

For the supplied PDE and pressure boundary conditions, adding a spatially constant value to `logk` multiplies permeability everywhere by the same positive factor. This leaves the pressure solution invariant while scaling Darcy flux. Accordingly, the controlled pressure regressors use `logk_amp` and `pattern_angle_deg`; all three parameters are retained when computing flux.

## Main findings

### Investigation 1 — sampling and initial surrogate comparison

- Validation selected POD-GPR with Latin hypercube sampling (LHS), 256 training samples, and validation mean relative error `1.026e-3` (SD `7.021e-5`).
- Validation selected the direct neural-network (NN) model with Sobol sampling, 256 training samples, and validation mean relative error `9.769e-3` (SD `2.430e-3`).
- On the held-out test set, POD-GPR (LHS) achieved global relative field error `9.580e-4`; the direct NN (Sobol) achieved `7.917e-3`.
- The sensitivity analysis concerned **domain-averaged pressure**, not flux. It indicated that pattern angle had the largest influence over the selected bounds; the amplitude had a smaller first-order effect and a larger total effect, consistent with interaction effects. `logk_mean` had negligible influence on this pressure quantity of interest.

**Interpretation limit:** the final POD-GPR and direct-NN models used different selected sampling strategies. Their held-out error difference therefore cannot be attributed to regression method alone. This motivated the controlled comparison in Investigation 2.

### Investigation 2 — controlled regression comparison and POD-rank sensitivity

At the common final comparison (`N=256`, 64 × 64 mesh, 32 POD modes, shared training/test designs and one shared POD basis), RBF-GPR had the lowest held-out pressure and flux errors among the six tested regression families:

| Regression family | Pressure global relative L2 | Flux relative L2 |
|---|---:|---:|
| **RBF-GPR** | **0.001004** | **0.01168** |
| Thin-plate spline | 0.01652 | 0.04099 |
| Multiquadric RBF | 0.02067 | 0.09248 |
| ANN | 0.02224 | 0.05372 |
| Tensor cubic B-spline | 0.05835 | 0.21285 |
| Polynomial (Ridge) | 0.07043 | 0.24905 |

For GPR at rank 32, the held-out pressure error decomposed into a POD truncation error of `9.936e-4` and a regression-only error of `1.442e-4`. Thus, the pressure error at this rank was dominated by POD truncation.

The fixed-RBF-GPR rank-sensitivity experiment used the same 256 training, 128 validation, and 256 test cases at every rank. The test pressure error fell from `1.572e-2` at rank 8 to approximately `1.57e-4` at rank 96. From ranks 96 to 255, additional modes produced negligible improvement: regression error became the dominant component. The validation errors at ranks 96–255 were also very close, so rank 96 is a practical knee for this particular 256-sample design, not a universally optimal rank.

See the [technical report](docs/technical_report.md) for methods, detailed results, qualifications, and figure placeholders.

## Repository structure

```text
src/
  darcy.py                         finite-volume Darcy solver
  sampling_surrogate_study.py      sampling + initial surrogate investigation
  experiment_a.py                  controlled regression-family comparison
  pod_rank_sensitivity.py          fixed-GPR POD-rank sensitivity
  smoke_test.py                    reduced software/workflow check

docs/
  technical_report.md              integrated research report
  limitations_and_reproducibility.md

results/
  initial_surrogate_investigation/ summary of reported results and data note
  regression_comparison/           held-out comparison table
  pod_rank_sensitivity/             rank-sensitivity CSV and summary

figures/                            add the selected plots here
```

## Reproducibility

Create an environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# Linux / macOS / Git Bash: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the investigations from the repository root (the scripts import modules from `src`, so set `PYTHONPATH`):

```bash
# Windows PowerShell
$env:PYTHONPATH = "src"
python src/sampling_surrogate_study.py
python src/experiment_a.py
python src/pod_rank_sensitivity.py

# Linux / macOS / Git Bash
export PYTHONPATH=src
python src/sampling_surrogate_study.py
python src/experiment_a.py
python src/pod_rank_sensitivity.py
```

The sampling study writes to `corrected_surrogate_results/`; the controlled regression study writes to `experiment_a_results/`; the rank study writes to `pod_rank_sensitivity_results/`. Run the smoke test before expensive runs if desired:

```bash
python src/smoke_test.py
```

The first study requires SALib for Sobol analysis. The code and configurations are preserved, but the original full raw output bundle for that investigation was not included in the materials used to assemble this repository; its reported headline values are summarized in the report. The regression-comparison and rank-sensitivity summary results are included under `results/`.

## Figures to add

Suggested figure filenames and placement notes are listed in [`figures/README.md`](figures/README.md). The report contains explicit image placeholders. Add plots generated by the full experiments—not smoke-test plots—before sharing the repository publicly.

## Limitations

- Parameter bounds are exploratory study bounds, not demonstrated physical operating limits.
- All surrogate results are conditional on the specified solver, 64 × 64 mesh, boundary conditions, and permeability parameterization.
- The initial sampling/model comparison is confounded by the different validation-selected sampling strategies.
- The controlled final regression comparison uses one common final training design and one held-out test design; it is not a multi-split estimate of test-performance variability.
- The rank-sensitivity study uses one fixed training/validation/test design. It identifies a rank-error trade-off for that design, not a universal optimum.
- The ANN reached its iteration limit in the final regression comparison; its result should be interpreted with that convergence warning.
- GPR interval coverage is reported in POD-coefficient space, not as pointwise pressure-field coverage.
- Mesh independence of surrogate errors has not been established by these experiments.

Further detail is provided in [`docs/limitations_and_reproducibility.md`](docs/limitations_and_reproducibility.md).
