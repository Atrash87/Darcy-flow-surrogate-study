"""Controlled POD-rank sensitivity study for the Darcy pressure surrogate.

Design:
- Reuses the same solver, parameter bounds, mesh, and sampling conventions as
  experiment_a.py.
- Uses one fixed 256-case training design, one fixed validation design, and one
  fixed untouched test design across every POD rank.
- Fits each POD basis using training snapshots only.
- Holds the regression family and kernel fixed to the RBF-GPR selected by
  Experiment A; GPR hyperparameters are fitted from training data only.
- Evaluates pressure error, POD truncation error, regression-only error, and
  flux error separately.

Run from this directory:
    python pod_rank_sensitivity.py

This is a follow-up sensitivity experiment, not a replacement for Experiment A.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import experiment_a as exp

# Predeclared ranks. The final value requests the maximum numerical rank of the
# centred training snapshot matrix (normally 255 for 256 snapshots).
POD_RANKS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 255]
OUTPUT_DIR = Path("pod_rank_sensitivity_results")
GPR_CONFIG = {"kernel_name": "RBF", "nu": None}


def fit_rank_limited_pod(snapshots: np.ndarray, rank: int) -> dict[str, Any]:
    """Fit a training-only centred POD basis with exactly `rank` modes."""
    started = time.perf_counter()
    mean_field = np.mean(snapshots, axis=0)
    centered = snapshots - mean_field
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= np.finfo(float).eps:
        raise RuntimeError("Training snapshots have no numerically meaningful variation.")
    tolerance = singular_values[0] * max(centered.shape) * np.finfo(float).eps
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < 1 or rank > numerical_rank:
        raise ValueError(f"Requested rank {rank}; numerical rank is {numerical_rank}.")

    energy = singular_values**2
    total_energy = float(np.sum(energy))
    phi = vt[:rank].T.copy()
    coefficients = centered @ phi
    captured = float(np.sum(energy[:rank]) / total_energy)
    return {
        "mean_field": mean_field,
        "phi": phi,
        "coefficients": coefficients,
        "rank": rank,
        "captured_energy": captured,
        "numerical_rank": numerical_rank,
        "singular_values": singular_values,
        "pod_fit_seconds": time.perf_counter() - started,
    }


def decomposition_error(
    truth: np.ndarray, projection: np.ndarray, prediction: np.ndarray
) -> tuple[float, float, float]:
    """Return truncation, regression-only, and identity-check errors."""
    trunc_sq = float(np.linalg.norm(truth - projection) ** 2)
    regression_sq = float(np.linalg.norm(projection - prediction) ** 2)
    total_sq = float(np.linalg.norm(truth - prediction) ** 2)
    identity = abs(total_sq - trunc_sq - regression_sq) / max(
        total_sq, np.finfo(float).tiny
    )
    return (
        exp.relative_l2(projection, truth),
        exp.relative_l2(prediction, projection),
        identity,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exp.validate_configuration()

    # These are deliberately the same deterministic designs used by the final
    # comparison in Experiment A. No data are regenerated between ranks.
    U_train = exp.make_lhs(exp.FINAL_N, exp.FINAL_TRAIN_SEED)
    U_validation = exp.make_lhs(exp.N_VALIDATION, exp.VALIDATION_SEED)
    U_test = exp.make_lhs(exp.N_TEST, exp.TEST_SEED)

    print("[data] solving fixed training, validation, and test designs")
    S_train, _, field_shape = exp.solve_many(U_train)
    S_validation, _, _ = exp.solve_many(U_validation)
    S_test, Q_test, _ = exp.solve_many(U_test)

    # Determine the actual numerical rank before validating the requested list.
    centered = S_train - np.mean(S_train, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    numerical_tolerance = singular_values[0] * max(centered.shape) * np.finfo(float).eps
    maximum_rank = int(np.count_nonzero(singular_values > numerical_tolerance))
    ranks = sorted(set(POD_RANKS))
    if any(rank > maximum_rank for rank in ranks):
        raise ValueError(
            f"POD_RANKS requests a rank above the numerical rank {maximum_rank}: {ranks}"
        )

    X_train = U_train[:, exp.PRESSURE_INPUT_COLUMNS]
    X_validation = U_validation[:, exp.PRESSURE_INPUT_COLUMNS]
    X_test = U_test[:, exp.PRESSURE_INPUT_COLUMNS]
    rows: list[dict[str, Any]] = []

    print(f"[design] N_train={len(U_train)}, N_validation={len(U_validation)}, "
          f"N_test={len(U_test)}, maximum numerical POD rank={maximum_rank}")
    print(f"[design] fixed regressor: RBF-GPR; ranks={ranks}")

    for rank in ranks:
        print(f"\n[rank {rank}] fitting POD basis and {rank} GPR coefficient models")
        pod = fit_rank_limited_pod(S_train, rank)
        C_validation, S_validation_projection = exp.project_fields(S_validation, pod)
        C_test, S_test_projection = exp.project_fields(S_test, pod)

        fit_started = time.perf_counter()
        predictor, gpr_fit_seconds, metadata = exp.fit_candidate(
            "GPR", GPR_CONFIG, X_train, pod["coefficients"],
            seed=exp.FINAL_TRAIN_SEED + 900_000 + rank,
        )
        C_validation_prediction = predictor(X_validation)
        S_validation_prediction = exp.reconstruct_fields(C_validation_prediction, pod)
        C_test_prediction = predictor(X_test)
        S_test_prediction = exp.reconstruct_fields(C_test_prediction, pod)
        prediction_seconds = time.perf_counter() - fit_started - gpr_fit_seconds

        validation_total = exp.field_metrics(S_validation_prediction, S_validation)
        validation_trunc, validation_regression, validation_identity = decomposition_error(
            S_validation, S_validation_projection, S_validation_prediction
        )
        test_total = exp.field_metrics(S_test_prediction, S_test)
        test_trunc, test_regression, test_identity = decomposition_error(
            S_test, S_test_projection, S_test_prediction
        )

        flux_started = time.perf_counter()
        Q_prediction = exp.predict_fluxes(S_test_prediction, U_test, field_shape)
        flux_seconds = time.perf_counter() - flux_started
        flux_scores = exp.flux_metrics(Q_prediction, Q_test)

        row = {
            "pod_rank": rank,
            "pod_numerical_rank": maximum_rank,
            "pod_captured_energy": pod["captured_energy"],
            "pod_fit_seconds": pod["pod_fit_seconds"],
            "gpr_kernel": "RBF",
            "n_gpr_coefficient_models": metadata["n_gp_models"],
            "gpr_fit_seconds": gpr_fit_seconds,
            "validation_prediction_seconds": "",
            "test_prediction_seconds": prediction_seconds,
            "test_prediction_ms_per_case": 1000.0 * prediction_seconds / len(U_test),
            "flux_postprocess_seconds": flux_seconds,
            "validation_field_rel_l2": validation_total["relative_l2_global"],
            "validation_pod_truncation_rel_l2": validation_trunc,
            "validation_regression_rel_l2_vs_projection": validation_regression,
            "validation_decomposition_identity_error": validation_identity,
            "test_field_rel_l2": test_total["relative_l2_global"],
            "test_field_case_mean_rel_l2": test_total["relative_l2_case_mean"],
            "test_field_case_max_rel_l2": test_total["relative_l2_case_max"],
            "test_pod_truncation_rel_l2": test_trunc,
            "test_regression_rel_l2_vs_projection": test_regression,
            "test_decomposition_identity_error": test_identity,
            **flux_scores,
        }
        rows.append(row)
        print(
            f"  energy={pod['captured_energy']:.8f} "
            f"test field={row['test_field_rel_l2']:.5e} "
            f"POD={test_trunc:.5e} regression={test_regression:.5e} "
            f"flux={flux_scores['flux_relative_l2']:.5e} "
            f"GPR fit={gpr_fit_seconds:.1f}s"
        )

    exp.write_csv(OUTPUT_DIR / "pod_rank_sensitivity.csv", rows)
    config = {
        "pod_ranks": ranks,
        "maximum_numerical_rank": maximum_rank,
        "training_size": len(U_train),
        "validation_size": len(U_validation),
        "test_size": len(U_test),
        "training_seed": exp.FINAL_TRAIN_SEED,
        "validation_seed": exp.VALIDATION_SEED,
        "test_seed": exp.TEST_SEED,
        "parameter_bounds": {name: [float(lo), float(hi)] for name, lo, hi in
                             zip(exp.PARAMETER_NAMES, exp.LOW, exp.HIGH)},
        "mesh": [exp.NX, exp.NY],
        "pressure_inputs": exp.PRESSURE_INPUT_NAMES,
        "regressor": "GPR with fixed RBF kernel family; hyperparameters fitted on training data",
        "selection_note": "No kernel/rank selection uses the held-out test set. The RBF family was selected in Experiment A using validation data at rank 32 and held fixed here.",
    }
    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # Plot field-error decomposition and flux error against retained rank.
    x = np.asarray([r["pod_rank"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(x, [r["test_field_rel_l2"] for r in rows], "o-", label="Total pressure error")
    ax.plot(x, [r["test_pod_truncation_rel_l2"] for r in rows], "s--", label="POD truncation")
    ax.plot(x, [r["test_regression_rel_l2_vs_projection"] for r in rows], "^--", label="Regression vs POD projection")
    ax.set_yscale("log")
    ax.set_xlabel("POD rank")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Held-out pressure error decomposition vs POD rank")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pressure_error_vs_pod_rank.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(x, [r["flux_relative_l2"] for r in rows], "o-", label="Flux relative L2 error")
    ax.set_yscale("log")
    ax.set_xlabel("POD rank")
    ax.set_ylabel("Flux relative L2 error")
    ax.set_title("Held-out flux error vs POD rank")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "flux_error_vs_pod_rank.png", dpi=180)
    plt.close(fig)

    summary_lines = [
        "POD-RANK SENSITIVITY — FIXED RBF-GPR",
        "====================================",
        f"Mesh: {exp.NX} x {exp.NY}",
        f"Training / validation / test: {len(U_train)} / {len(U_validation)} / {len(U_test)}",
        f"Numerical rank of centred training snapshots: {maximum_rank}",
        "Same training, validation and test designs are reused at every rank.",
        "POD basis is fitted on training snapshots only.",
        "RBF-GPR family is fixed; GP hyperparameters are fitted using training data only.",
        "Validation and test sets are not used to fit the POD basis or GP models.",
        "The held-out test set is not used to select a rank or kernel.",
        "",
        "rank | captured energy | test pressure | POD truncation | regression | test flux",
    ]
    for r in rows:
        summary_lines.append(
            f"{r['pod_rank']:4d} | {r['pod_captured_energy']:.8f} | "
            f"{r['test_field_rel_l2']:.6e} | {r['test_pod_truncation_rel_l2']:.6e} | "
            f"{r['test_regression_rel_l2_vs_projection']:.6e} | {r['flux_relative_l2']:.6e}"
        )
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\n[complete] results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
