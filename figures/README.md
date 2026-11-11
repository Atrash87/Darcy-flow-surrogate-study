# Figure placement guide

Add figures generated from the **full experiments**, not smoke-test runs. Keep the filenames below so the report image links resolve automatically.

| Filename | Suggested content | Source |
|---|---|---|
| `investigation1_validation_error_vs_samples.png` | Validation error versus training size for sampling strategies and the two initial surrogate families | Sampling investigation |
| `investigation1_sobol_indices.png` | First- and total-order Sobol indices for domain-averaged pressure | Sampling investigation |
| `investigation2_pressure_parity.png` | Held-out true versus predicted pressure for the six regression families | Controlled regression comparison |
| `investigation2_pressure_error_map.png` | Spatial pressure error for the final GPR model or representative test cases | Controlled regression comparison |
| `pod_rank_pressure_error_decomposition.png` | Total pressure error, POD truncation error, and regression-only error versus POD rank (log scale) | Rank-sensitivity experiment |
| `pod_rank_flux_error.png` | Held-out flux relative error versus POD rank | Rank-sensitivity experiment |

The report uses relative paths from `docs/technical_report.md` to this folder.
