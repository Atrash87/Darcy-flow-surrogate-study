# Limitations and reproducibility notes

This note is intended to travel with the project so that the results are not presented more strongly than the experiments support.

## What is controlled

- In the controlled regression comparison, all six regression families use the same final 256-case training design, the same training-only POD basis, and the same 256-case held-out test design.
- Validation is used for candidate selection; the final test set is not used for hyperparameter selection.
- In the POD-rank sweep, the same 256 training, 128 validation, and 256 test cases are reused at every rank. POD bases are fitted on training snapshots only.
- Pressure and flux are evaluated separately. Pressure regression uses `logk_amp` and `pattern_angle_deg`; flux post-processing uses all three parameters.

## What is not established

- The initial POD-GPR versus direct-NN held-out comparison is confounded by different selected sampling strategies (LHS versus Sobol).
- One held-out split does not quantify the distribution of test errors across independent data designs.
- The rank sweep does not establish a universal optimal rank. It uses one fixed design and one regression family.
- The high-rank regression floor has not been diagnosed as a training-data limitation versus a model limitation.
- The ANN did not converge within the configured iteration budget in the reported final comparison.
- GPR interval coverage was measured in coefficient space and was below nominal coverage in the reported run.
- The study does not establish mesh-independent surrogate accuracy or validate the parameter bounds for a specific geological application.

## Reproduction checklist

Before publishing or sending this repository:

1. Confirm that `src/darcy.py` is the exact solver version used to generate the reported results.
2. Preserve the random seeds and configuration values in each script.
3. Add the original full output bundle for the sampling investigation (validation records, final test table, sensitivity indices, configuration, and saved data archive) if available.
4. Add the full-run outputs from the controlled regression comparison, not smoke-test outputs.
5. Add the full-run rank-sensitivity CSV and summary; these are included in this repository.
6. Add figures generated from the full experiments and use the filenames referenced in the report.
7. If any configuration or code is changed, rerun the relevant experiment and update the report rather than retaining old result values.

## Smoke tests

Smoke-test runs are for checking code paths and output generation. Their coarse meshes and small sample sizes are not research results and must not be mixed with the full-run tables in the report.
