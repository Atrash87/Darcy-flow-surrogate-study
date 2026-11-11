# Initial sampling and surrogate investigation — reported summary

The code is preserved at `src/sampling_surrogate_study.py`. The full original run output bundle was not present among the files used to assemble this repository. The values below are carried over from the previously reported experiment and are included for transparent summary only.

- Validation-selected POD-GPR: LHS, N=256, validation mean relative L2 = 1.025979e-3, SD = 7.021016e-5.
- Validation-selected direct NN: Sobol, N=256, validation mean relative L2 = 9.768674e-3, SD = 2.429706e-3.
- Held-out POD-GPR: global relative L2 = 9.580484e-4; mean case = 8.601140e-4; max case = 3.755522e-3.
- Held-out direct NN: global relative L2 = 7.916819e-3; mean case = 6.856115e-3; max case = 3.213562e-2.

The final models used different selected sampling strategies (LHS for POD-GPR and Sobol for direct NN). The final error difference is therefore not an isolated comparison of regression algorithms.

The reported Sobol sensitivity analysis used domain-averaged pressure as its quantity of interest. Its indices and qualifications are documented in `docs/technical_report.md`.

**Before public release:** add the original `corrected_surrogate_results/` output directory if available, including its CSV files, configuration, summary, plots, and NPZ archive.
