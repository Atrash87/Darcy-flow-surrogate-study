"""Experiment A: compare regression methods on one shared POD representation.

All regressors for a given training design receive:
  * the exact same training parameter samples and pressure snapshots;
  * the exact same POD mean field, basis and retained rank;
  * the exact same validation set.

The final test set is generated independently and evaluated only after the
validation-based candidate selection is complete.

Important structural property of the supplied Darcy model
---------------------------------------------------------
For -div(k grad(p)) = 0 with fixed-pressure and no-flow boundaries, multiplying
k everywhere by exp(logk_mean) does not change p. Thus pressure depends on
(logk_amp, pattern_angle_deg), while q scales with logk_mean. By default, the
pressure regressors use columns (1, 2) of the normalized three-parameter input.
Flux is evaluated with all three physical parameters. Set
PRESSURE_INPUT_COLUMNS = (0, 1, 2) only if you deliberately want to retain the
inactive input dimension as a separate diagnostic experiment.

Run from this directory:
    python -m pip install -r requirements.txt
    python experiment_a.py
"""
from __future__ import annotations

import csv
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from darcy import boundary_fluxes, solve_darcy


# =============================================================================
# 1. USER CONFIGURATION — REVIEW THESE VALUES BEFORE A FULL RUN
# =============================================================================

# These are the bounds in your previous study script. They are not independently
# established physical limits; replace them if your intended parameter domain differs.
LOW = np.array([-1.0, 0.2, 0.0], dtype=float)
HIGH = np.array([1.0, 1.5, 180.0], dtype=float)
PARAMETER_NAMES = ["logk_mean", "logk_amp", "pattern_angle_deg"]

# Main field surrogate: only parameters that change pressure in this model.
# Use (0, 1, 2) for a deliberately redundant 3-input diagnostic comparison.
PRESSURE_INPUT_COLUMNS = (1, 2)
PRESSURE_INPUT_NAMES = [PARAMETER_NAMES[i] for i in PRESSURE_INPUT_COLUMNS]

# Grid and boundary pressures. Match these to the solver used to generate data.
NX = 64
NY = 64
P_LEFT = 1.0
P_RIGHT = 0.0

# POD retains at least 1 - POD_TOL of the centred snapshot energy.
POD_TOL = 1e-4

# Validation study. Every size uses an independently generated LHS design.
SAMPLE_SIZES = [16, 32, 64, 128, 256]
N_REPEATS = 3
FINAL_N = 256  # align the final head-to-head budget with the previous 256-snapshot study
N_VALIDATION = 128
N_TEST = 256

# Regression training settings.
GPR_ALPHA = 1e-8
GPR_RESTARTS = 1  # one additional optimizer restart for more robust kernel tuning
GPR_LENGTH_SCALE_BOUNDS = (1e-2, 1e2)
ANN_MAX_ITER = 2500
ANN_TOL = 1e-5

# Seeds make designs and stochastic ANN fits reproducible.
MASTER_SEED = 20261111
VALIDATION_SEED = 11011
TEST_SEED = 22022
FINAL_TRAIN_SEED = 33033

OUTPUT_DIR = Path("experiment_a_results")


# =============================================================================
# 2. CONFIGURATION, DATA GENERATION AND METRICS
# =============================================================================


def validate_configuration() -> None:
    if LOW.shape != HIGH.shape or LOW.ndim != 1 or len(LOW) != 3:
        raise ValueError("LOW and HIGH must be length-3 vectors.")
    if np.any(HIGH <= LOW):
        raise ValueError("Every upper parameter bound must exceed its lower bound.")
    if any(n < 8 for n in SAMPLE_SIZES):
        raise ValueError("Use training sample sizes of at least 8 for this comparison.")
    if sorted(set(SAMPLE_SIZES)) != SAMPLE_SIZES:
        raise ValueError("SAMPLE_SIZES must be unique and sorted.")
    if FINAL_N not in SAMPLE_SIZES:
        raise ValueError("FINAL_N must be included in SAMPLE_SIZES.")
    if N_REPEATS < 2:
        raise ValueError("Use at least two independent training-design repetitions.")
    if N_VALIDATION < 16 or N_TEST < 16:
        raise ValueError("Validation and test sets should each contain at least 16 cases.")
    if not (0.0 < POD_TOL < 1.0):
        raise ValueError("POD_TOL must be between zero and one.")
    if not PRESSURE_INPUT_COLUMNS:
        raise ValueError("At least one pressure input column is required.")
    if any(i not in (0, 1, 2) for i in PRESSURE_INPUT_COLUMNS):
        raise ValueError("PRESSURE_INPUT_COLUMNS must contain indices from 0, 1, 2.")
    if len(set(PRESSURE_INPUT_COLUMNS)) != len(PRESSURE_INPUT_COLUMNS):
        raise ValueError("PRESSURE_INPUT_COLUMNS cannot contain duplicates.")


def to_physical(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    return LOW + U * (HIGH - LOW)


def make_lhs(n: int, seed: int) -> np.ndarray:
    """Generate a parameter design with a true LHS in pressure-active space.

    With PRESSURE_INPUT_COLUMNS=(1, 2), logk_amp and angle are sampled jointly
    by a 2D LHS. The pressure-inactive logk_mean is sampled independently by a
    1D LHS, so it remains available for flux diagnostics without degrading the
    active 2D design. If the user deliberately enables the 3-input ablation,
    all three inputs are generated jointly as a 3D LHS instead.
    """
    active = tuple(PRESSURE_INPUT_COLUMNS)
    inactive = tuple(i for i in range(3) if i not in active)
    U = np.empty((n, 3), dtype=float)
    U[:, active] = qmc.LatinHypercube(d=len(active), seed=seed).random(n)
    for offset, column in enumerate(inactive, start=1):
        U[:, column] = qmc.LatinHypercube(d=1, seed=seed + 1000 * offset).random(n).ravel()
    return U


def verify_parameter_structure() -> dict[str, float]:
    """Check the two structural properties used by this experiment.

    (1) Pressure is invariant to a global log-permeability shift.
    (2) The supplied heterogeneity pattern does not make the fields at the two
        angle bounds equivalent. This supports a bounded, non-periodic input.
    """
    amp = 0.5 * (LOW[1] + HIGH[1])
    angle = 0.37 * LOW[2] + 0.63 * HIGH[2]
    mean_low = tuple(float(v) for v in (LOW[0], amp, angle))
    mean_high = tuple(float(v) for v in (HIGH[0], amp, angle))
    p_low, _, _, q_low = solve_darcy(
        nx=NX, ny=NY, params=mean_low, p_left=P_LEFT, p_right=P_RIGHT
    )
    p_high, _, _, q_high = solve_darcy(
        nx=NX, ny=NY, params=mean_high, p_left=P_LEFT, p_right=P_RIGHT
    )
    pressure_mean_invariance = relative_l2(p_low.ravel()[None, :], p_high.ravel()[None, :])
    expected_flux_ratio = float(np.exp(HIGH[0] - LOW[0]))
    observed_flux_ratio = float(q_high / q_low)
    flux_scale_relative_error = abs(observed_flux_ratio - expected_flux_ratio) / abs(expected_flux_ratio)

    mid_mean = 0.5 * (LOW[0] + HIGH[0])
    p_angle_low, _, _, _ = solve_darcy(
        nx=NX, ny=NY,
        params=(float(mid_mean), float(amp), float(LOW[2])),
        p_left=P_LEFT, p_right=P_RIGHT,
    )
    p_angle_high, _, _, _ = solve_darcy(
        nx=NX, ny=NY,
        params=(float(mid_mean), float(amp), float(HIGH[2])),
        p_left=P_LEFT, p_right=P_RIGHT,
    )
    endpoint_pressure_difference = relative_l2(
        p_angle_low.ravel()[None, :], p_angle_high.ravel()[None, :]
    )

    checks = {
        "pressure_relative_difference_across_logk_mean_bounds": float(pressure_mean_invariance),
        "expected_flux_ratio_high_vs_low_logk_mean": expected_flux_ratio,
        "observed_flux_ratio_high_vs_low_logk_mean": observed_flux_ratio,
        "flux_ratio_relative_error": float(flux_scale_relative_error),
        "pressure_relative_difference_between_angle_bounds": float(endpoint_pressure_difference),
    }
    if pressure_mean_invariance > 1e-9:
        raise RuntimeError(
            "Pressure is not invariant to logk_mean at the expected numerical tolerance. "
            "Recheck the solver, equation, or boundary conditions before excluding this input."
        )
    if flux_scale_relative_error > 1e-8:
        raise RuntimeError(
            "Flux does not scale as exp(delta logk_mean). Recheck the solver before proceeding."
        )
    if endpoint_pressure_difference < 1e-8:
        raise RuntimeError(
            "The angle-bound fields are numerically indistinguishable. The current "
            "non-periodic angle interpretation should be reviewed."
        )
    return checks


def solve_many(U: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Run the full-order Darcy model; return flattened fields, left fluxes, shape."""
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[1] != 3:
        raise ValueError("Input design must have shape (n_samples, 3).")

    snapshots: list[np.ndarray] = []
    fluxes: list[float] = []
    field_shape: tuple[int, int] | None = None

    for i, u in enumerate(U):
        params = tuple(float(v) for v in to_physical(u))
        pressure, _, _, q_left = solve_darcy(
            nx=NX, ny=NY, params=params, p_left=P_LEFT, p_right=P_RIGHT
        )
        pressure = np.asarray(pressure, dtype=float)
        if pressure.shape != (NY, NX):
            raise ValueError(
                f"Solver returned {pressure.shape}; expected {(NY, NX)}. "
                "Check NX/NY against your solver configuration."
            )
        if not np.all(np.isfinite(pressure)) or not np.isfinite(q_left):
            raise ValueError(f"Non-finite solver result at sample {i}.")
        if field_shape is None:
            field_shape = pressure.shape
        elif pressure.shape != field_shape:
            raise ValueError("Inconsistent pressure-field shapes from solver.")
        snapshots.append(pressure.ravel())
        fluxes.append(float(q_left))

    assert field_shape is not None
    return np.vstack(snapshots), np.asarray(fluxes), field_shape


def relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    denom = max(float(np.linalg.norm(truth)), np.finfo(float).tiny)
    return float(np.linalg.norm(pred - truth) / denom)


def field_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("Field prediction and truth must have matching (cases, dofs) shapes.")
    tiny = np.finfo(float).tiny
    case_num = np.linalg.norm(pred - truth, axis=1)
    case_den = np.maximum(np.linalg.norm(truth, axis=1), tiny)
    errors = case_num / case_den
    return {
        "relative_l2_global": relative_l2(pred, truth),
        "relative_l2_case_mean": float(np.mean(errors)),
        "relative_l2_case_median": float(np.median(errors)),
        "relative_l2_case_max": float(np.max(errors)),
        "rmse_all_values": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "max_absolute_error": float(np.max(np.abs(pred - truth))),
    }


def flux_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    tiny = np.finfo(float).tiny
    rel = np.abs(pred - truth) / np.maximum(np.abs(truth), tiny)
    return {
        "flux_relative_l2": relative_l2(pred, truth),
        "flux_relative_abs_error_mean": float(np.mean(rel)),
        "flux_relative_abs_error_median": float(np.median(rel)),
        "flux_relative_abs_error_max": float(np.max(rel)),
        "flux_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # Use the union of fields, since optional uncertainty metrics are blank for non-GP models.
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# 3. TRAINING-ONLY POD BASIS
# =============================================================================


def fit_pod(U_train: np.ndarray, S_train: np.ndarray) -> dict[str, Any]:
    """Fit a centred POD basis using training snapshots only."""
    del U_train  # kept in the signature to make the training-only boundary explicit
    started = time.perf_counter()
    mean_field = np.mean(S_train, axis=0)
    centered = S_train - mean_field
    _, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
    energy = singular_values**2
    total_energy = float(np.sum(energy))

    if total_energy <= np.finfo(float).eps:
        phi = np.zeros((S_train.shape[1], 0), dtype=float)
        rank = 0
        captured = 1.0
    else:
        cumulative = np.cumsum(energy) / total_energy
        rank = int(np.searchsorted(cumulative, 1.0 - POD_TOL) + 1)
        rank = max(1, min(rank, Vt.shape[0]))
        phi = Vt[:rank].T.copy()
        captured = float(cumulative[rank - 1])

    coefficients = centered @ phi
    return {
        "mean_field": mean_field,
        "phi": phi,
        "coefficients": coefficients,
        "rank": rank,
        "captured_energy": captured,
        "pod_fit_seconds": time.perf_counter() - started,
    }


def project_fields(S: np.ndarray, pod: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return POD coefficients and orthogonal projection of fields onto the basis."""
    mean = pod["mean_field"]
    phi = pod["phi"]
    if phi.shape[1] == 0:
        coeff = np.zeros((len(S), 0), dtype=float)
        projected = np.repeat(mean[None, :], len(S), axis=0)
    else:
        coeff = (S - mean) @ phi
        projected = mean + coeff @ phi.T
    return coeff, projected


def reconstruct_fields(coefficients: np.ndarray, pod: dict[str, Any]) -> np.ndarray:
    mean = pod["mean_field"]
    phi = pod["phi"]
    if phi.shape[1] == 0:
        return np.repeat(mean[None, :], len(coefficients), axis=0)
    return mean + coefficients @ phi.T


# =============================================================================
# 4. REGRESSION CANDIDATES
# =============================================================================


def build_candidates() -> list[tuple[str, dict[str, Any]]]:
    """Return (family, hyperparameter configuration) pairs."""
    candidates: list[tuple[str, dict[str, Any]]] = []

    for kernel_name, nu in [("RBF", None), ("Matern_0p5", 0.5),
                            ("Matern_1p5", 1.5), ("Matern_2p5", 2.5)]:
        candidates.append(("GPR", {"kernel_name": kernel_name, "nu": nu}))

    for epsilon in (0.5, 1.0, 2.0, 4.0):
        candidates.append(("RBF_Multiquadric", {"epsilon": epsilon, "smoothing": 1e-6}))

    for smoothing in (1e-8, 1e-5, 1e-3, 1e-1):
        candidates.append(("ThinPlateSpline", {"smoothing": smoothing}))

    for n_knots in (3, 4):
        for alpha in (1e-4, 1e-2):
            candidates.append(("TensorCubicBSpline", {"n_knots": n_knots, "alpha": alpha}))

    for hidden in ((32, 32), (64, 64)):
        for alpha in (1e-4, 1e-3):
            candidates.append(("ANN", {"hidden_layer_sizes": list(hidden), "alpha": alpha}))

    for degree in (1, 2):
        for alpha in (1e-6, 1e-3):
            candidates.append(("Polynomial", {"degree": degree, "alpha": alpha}))

    return candidates


def tensor_bspline_features(
    X: np.ndarray,
    transformers: list[SplineTransformer] | None = None,
    n_knots: int = 3,
) -> tuple[np.ndarray, list[SplineTransformer]]:
    """Construct a tensor-product cubic B-spline basis for 2D or 3D inputs."""
    if transformers is None:
        transformers = [
            SplineTransformer(
                degree=3,
                knots=np.linspace(0.0, 1.0, n_knots)[:, None],
                extrapolation="constant",
                include_bias=True,
            )
            for _ in range(X.shape[1])
        ]
        one_d = [transformers[j].fit_transform(X[:, j:j + 1]) for j in range(X.shape[1])]
    else:
        one_d = [transformers[j].transform(X[:, j:j + 1]) for j in range(X.shape[1])]

    # Tensor products for arbitrary dimension; our current problem has two active inputs.
    product = one_d[0]
    for basis in one_d[1:]:
        product = (product[:, :, None] * basis[:, None, :]).reshape(len(X), -1)
    return product, transformers


def fit_candidate(
    family: str,
    config: dict[str, Any],
    X_train: np.ndarray,
    C_train: np.ndarray,
    seed: int,
) -> tuple[Callable[..., Any], float, dict[str, Any]]:
    """Fit one regression candidate to POD coefficients."""
    started = time.perf_counter()
    target_scaler = StandardScaler().fit(C_train)
    C_scaled = target_scaler.transform(C_train)
    metadata: dict[str, Any] = {}

    if family == "GPR":
        gps = []
        for k in range(C_scaled.shape[1]):
            if config["kernel_name"] == "RBF":
                base_kernel = RBF(
                    length_scale=np.full(X_train.shape[1], 0.3),
                    length_scale_bounds=GPR_LENGTH_SCALE_BOUNDS,
                )
            else:
                nu = float(config["nu"])
                base_kernel = Matern(
                    length_scale=np.full(X_train.shape[1], 0.3),
                    length_scale_bounds=GPR_LENGTH_SCALE_BOUNDS,
                    nu=nu,
                )
            kernel = ConstantKernel(1.0, (1e-2, 1e2)) * base_kernel
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=GPR_ALPHA,
                normalize_y=False,
                n_restarts_optimizer=GPR_RESTARTS,
                random_state=seed + k,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(X_train, C_scaled[:, k])
            gps.append(gp)

        def predict(X_query: np.ndarray, return_std: bool = False):
            means, stds = [], []
            for gp in gps:
                if return_std:
                    mean_k, std_k = gp.predict(X_query, return_std=True)
                    means.append(mean_k)
                    stds.append(std_k)
                else:
                    means.append(gp.predict(X_query))
            mean_scaled = np.column_stack(means)
            mean = target_scaler.inverse_transform(mean_scaled)
            if return_std:
                std_scaled = np.column_stack(stds)
                std = std_scaled * target_scaler.scale_[None, :]
                return mean, std
            return mean

        metadata["n_gp_models"] = len(gps)

    elif family == "RBF_Multiquadric":
        model = RBFInterpolator(
            X_train,
            C_scaled,
            kernel="multiquadric",
            epsilon=float(config["epsilon"]),
            smoothing=float(config["smoothing"]),
        )

        def predict(X_query: np.ndarray, return_std: bool = False):
            if return_std:
                raise ValueError("RBF interpolation does not provide calibrated predictive uncertainty.")
            return target_scaler.inverse_transform(np.asarray(model(X_query)))

    elif family == "ThinPlateSpline":
        model = RBFInterpolator(
            X_train,
            C_scaled,
            kernel="thin_plate_spline",
            degree=1,
            smoothing=float(config["smoothing"]),
        )

        def predict(X_query: np.ndarray, return_std: bool = False):
            if return_std:
                raise ValueError("Thin-plate spline does not provide predictive uncertainty.")
            return target_scaler.inverse_transform(np.asarray(model(X_query)))

    elif family == "TensorCubicBSpline":
        n_knots = int(config["n_knots"])
        basis_train, transformers = tensor_bspline_features(X_train, n_knots=n_knots)
        # Keep the bounded partition-of-unity basis unscaled. Standardizing rare
        # boundary basis functions can magnify extrapolation dramatically.
        model = Ridge(alpha=float(config["alpha"]), fit_intercept=False)
        model.fit(basis_train, C_scaled)

        def predict(X_query: np.ndarray, return_std: bool = False):
            if return_std:
                raise ValueError("B-spline regression does not provide predictive uncertainty.")
            B_query, _ = tensor_bspline_features(X_query, transformers=transformers)
            return target_scaler.inverse_transform(model.predict(B_query))

        metadata["spline_feature_count"] = int(basis_train.shape[1])

    elif family == "ANN":
        input_scaler = StandardScaler().fit(X_train)
        X_scaled = input_scaler.transform(X_train)
        model = MLPRegressor(
            hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
            activation="tanh",
            solver="adam",
            alpha=float(config["alpha"]),
            learning_rate_init=1e-3,
            max_iter=ANN_MAX_ITER,
            early_stopping=False,
            n_iter_no_change=150,
            tol=ANN_TOL,
            random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_scaled, C_scaled)
        metadata["ann_n_iter"] = int(model.n_iter_)
        metadata["ann_convergence_warning"] = bool(
            any(issubclass(w.category, ConvergenceWarning) for w in caught)
        )

        def predict(X_query: np.ndarray, return_std: bool = False):
            if return_std:
                raise ValueError("ANN in this experiment does not provide calibrated uncertainty.")
            return target_scaler.inverse_transform(model.predict(input_scaler.transform(X_query)))

    elif family == "Polynomial":
        degree = int(config["degree"])
        polynomial = PolynomialFeatures(degree=degree, include_bias=False)
        F_train = polynomial.fit_transform(X_train)
        feature_scaler = StandardScaler().fit(F_train)
        F_scaled = feature_scaler.transform(F_train)
        model = Ridge(alpha=float(config["alpha"]), fit_intercept=True)
        model.fit(F_scaled, C_scaled)

        def predict(X_query: np.ndarray, return_std: bool = False):
            if return_std:
                raise ValueError("Polynomial regression does not provide predictive uncertainty.")
            F_query = polynomial.transform(X_query)
            return target_scaler.inverse_transform(model.predict(feature_scaler.transform(F_query)))

        metadata["polynomial_feature_count"] = int(F_train.shape[1])

    else:
        raise ValueError(f"Unknown regression family: {family}")

    return predict, time.perf_counter() - started, metadata


# =============================================================================
# 5. VALIDATION-ONLY MODEL AND HYPERPARAMETER COMPARISON
# =============================================================================


def evaluate_validation_candidate(
    family: str,
    config: dict[str, Any],
    U_train: np.ndarray,
    pod: dict[str, Any],
    U_validation: np.ndarray,
    S_validation: np.ndarray,
    repeat: int,
    seed: int,
) -> dict[str, Any]:
    C_validation, S_validation_projected = project_fields(S_validation, pod)
    X_train = U_train[:, PRESSURE_INPUT_COLUMNS]
    X_validation = U_validation[:, PRESSURE_INPUT_COLUMNS]

    candidate_id = json.dumps(config, sort_keys=True)
    base = {
        "repeat": repeat,
        "n_train": len(U_train),
        "model_family": family,
        "candidate_id": candidate_id,
        "config": candidate_id,
        "pod_rank": pod["rank"],
        "pod_captured_energy": pod["captured_energy"],
        "pod_fit_seconds": pod["pod_fit_seconds"],
    }

    if pod["rank"] == 0:
        pred_fields = np.repeat(pod["mean_field"][None, :], len(U_validation), axis=0)
        total = field_metrics(pred_fields, S_validation)
        truncation = field_metrics(S_validation_projected, S_validation)
        return {
            **base,
            **{f"validation_{k}": v for k, v in total.items()},
            "validation_regression_rel_l2": 0.0,
            "validation_pod_truncation_rel_l2": truncation["relative_l2_global"],
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "uncertainty_predict_seconds": "",
            "coverage90_coeff": "",
            "coverage95_coeff": "",
            "mean_width90_coeff": "",
            "mean_width95_coeff": "",
            "status": "ok_constant_field",
        }

    predictor, fit_seconds, metadata = fit_candidate(
        family, config, X_train, pod["coefficients"], seed
    )
    # Time mean inference consistently for every method, including POD-field
    # reconstruction. GP uncertainty is timed separately so it does not make
    # GPR mean prediction look slower than deterministic regressors.
    predict_started = time.perf_counter()
    C_pred = predictor(X_validation)
    S_pred = reconstruct_fields(C_pred, pod)
    predict_seconds = time.perf_counter() - predict_started

    C_std = None
    uncertainty_predict_seconds: float | str = ""
    if family == "GPR":
        uncertainty_started = time.perf_counter()
        _, C_std = predictor(X_validation, return_std=True)
        uncertainty_predict_seconds = time.perf_counter() - uncertainty_started
    total = field_metrics(S_pred, S_validation)
    regression = relative_l2(S_pred, S_validation_projected)
    truncation = field_metrics(S_validation_projected, S_validation)

    coverage90 = coverage95 = width90 = width95 = ""
    if C_std is not None:
        z90, z95 = 1.6448536269514722, 1.959963984540054
        abs_resid = np.abs(C_validation - C_pred)
        safe_std = np.maximum(C_std, 1e-14)
        coverage90 = float(np.mean(abs_resid <= z90 * safe_std))
        coverage95 = float(np.mean(abs_resid <= z95 * safe_std))
        width90 = float(np.mean(2.0 * z90 * C_std))
        width95 = float(np.mean(2.0 * z95 * C_std))

    return {
        **base,
        **{f"validation_{k}": v for k, v in total.items()},
        "validation_regression_rel_l2": regression,
        "validation_pod_truncation_rel_l2": truncation["relative_l2_global"],
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "uncertainty_predict_seconds": uncertainty_predict_seconds,
        "coverage90_coeff": coverage90,
        "coverage95_coeff": coverage95,
        "mean_width90_coeff": width90,
        "mean_width95_coeff": width95,
        "status": "ok",
        **metadata,
    }


def run_validation_experiment(
    U_validation: np.ndarray,
    S_validation: np.ndarray,
) -> list[dict[str, Any]]:
    candidates = build_candidates()
    records: list[dict[str, Any]] = []
    total_runs = N_REPEATS * len(SAMPLE_SIZES)
    run_no = 0

    for repeat in range(N_REPEATS):
        for n_train in SAMPLE_SIZES:
            run_no += 1
            seed = MASTER_SEED + repeat * 100_000 + n_train * 10
            U_train = make_lhs(n_train, seed)
            S_train, _, _ = solve_many(U_train)
            print(
                f"\n[validation {run_no}/{total_runs}] repeat={repeat + 1}/{N_REPEATS}, "
                f"N={n_train}; fitting one shared POD basis"
            )
            pod = fit_pod(U_train, S_train)

            for candidate_index, (family, config) in enumerate(candidates, start=1):
                try:
                    row = evaluate_validation_candidate(
                        family, config, U_train, pod,
                        U_validation, S_validation,
                        repeat=repeat, seed=seed + candidate_index,
                    )
                except Exception as exc:
                    row = {
                        "repeat": repeat,
                        "n_train": n_train,
                        "model_family": family,
                        "candidate_id": json.dumps(config, sort_keys=True),
                        "config": json.dumps(config, sort_keys=True),
                        "status": f"failed: {type(exc).__name__}: {exc}",
                        "validation_relative_l2_global": np.nan,
                    }
                records.append(row)
                score = row.get("validation_relative_l2_global", np.nan)
                if np.isfinite(score):
                    print(f"  {family:22s} {str(config):48s} val={score:.4e}")
                else:
                    print(f"  {family:22s} {str(config):48s} FAILED: {row.get('status', '')}")

    return records


def summarize_validation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [r for r in records if r.get("status") == "ok"]
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in successful:
        key = (row["model_family"], int(row["n_train"]), row["candidate_id"])
        groups.setdefault(key, []).append(row)

    candidate_summary: list[dict[str, Any]] = []
    for (family, n_train, candidate_id), rows in groups.items():
        values = np.asarray([r["validation_relative_l2_global"] for r in rows], dtype=float)
        reg_values = np.asarray([r["validation_regression_rel_l2"] for r in rows], dtype=float)
        fit_values = np.asarray([r["fit_seconds"] for r in rows], dtype=float)
        pred_values = np.asarray([r["predict_seconds"] for r in rows], dtype=float)
        # Do not rank a candidate on a subset of repetitions: a failed fit in
        # one repeat must not give it an artificially favourable mean.
        if len(rows) != N_REPEATS:
            continue
        candidate_summary.append({
            "model_family": family,
            "n_train": n_train,
            "candidate_id": candidate_id,
            "n_successful_repeats": len(rows),
            "validation_rel_l2_mean": float(np.mean(values)),
            "validation_rel_l2_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "validation_regression_rel_l2_mean": float(np.mean(reg_values)),
            "fit_seconds_mean": float(np.mean(fit_values)),
            "predict_seconds_mean": float(np.mean(pred_values)),
            "uncertainty_predict_seconds_mean": _finite_mean([
                r.get("uncertainty_predict_seconds") for r in rows
            ]),
            "pod_rank_mean": float(np.mean([r["pod_rank"] for r in rows])),
            "pod_truncation_rel_l2_mean": float(np.mean([
                r["validation_pod_truncation_rel_l2"] for r in rows
            ])),
            "coverage90_coeff_mean": _finite_mean([r.get("coverage90_coeff") for r in rows]),
            "coverage95_coeff_mean": _finite_mean([r.get("coverage95_coeff") for r in rows]),
            "mean_width90_coeff": _finite_mean([r.get("mean_width90_coeff") for r in rows]),
            "mean_width95_coeff": _finite_mean([r.get("mean_width95_coeff") for r in rows]),
            "config": candidate_id,
        })

    # At each training size, pick the hyperparameter candidate for each family
    # using the mean validation error across independent design repetitions.
    best_by_size: list[dict[str, Any]] = []
    families_sizes = sorted({(r["model_family"], r["n_train"]) for r in candidate_summary})
    for family, n_train in families_sizes:
        candidates_here = [
            r for r in candidate_summary
            if r["model_family"] == family and r["n_train"] == n_train
        ]
        best = min(candidates_here, key=lambda r: r["validation_rel_l2_mean"])
        best_by_size.append({**best, "selection": "best candidate within family at this N"})

    return best_by_size


def _finite_mean(values: list[Any]) -> float | str:
    numeric = [float(v) for v in values if isinstance(v, (int, float, np.number)) and np.isfinite(v)]
    return float(np.mean(numeric)) if numeric else ""


def select_final_candidates(summary: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    at_final_n = [r for r in summary if int(r["n_train"]) == FINAL_N]
    selected: dict[str, dict[str, Any]] = {}
    for family in sorted({r["model_family"] for r in at_final_n}):
        rows = [r for r in at_final_n if r["model_family"] == family]
        if not rows:
            continue
        selected[family] = min(rows, key=lambda r: r["validation_rel_l2_mean"])
    return selected


# =============================================================================
# 6. FINAL COMMON-BASIS TEST — TEST SET USED ONCE
# =============================================================================


def predict_fluxes(
    pressure_fields: np.ndarray,
    U: np.ndarray,
    field_shape: tuple[int, int],
) -> np.ndarray:
    result = np.empty(len(U), dtype=float)
    for i, (field, u) in enumerate(zip(pressure_fields, U)):
        params = tuple(float(v) for v in to_physical(u))
        q_left, _ = boundary_fluxes(
            field.reshape(field_shape), params,
            nx=NX, ny=NY, p_left=P_LEFT, p_right=P_RIGHT,
        )
        result[i] = q_left
    return result


def run_final_test(
    selected: dict[str, dict[str, Any]],
    U_test: np.ndarray,
    S_test: np.ndarray,
    Q_test: np.ndarray,
    field_shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    """Fit every selected family on one dataset and one shared POD basis."""
    U_final = make_lhs(FINAL_N, FINAL_TRAIN_SEED)
    S_final, Q_final, _ = solve_many(U_final)
    pod = fit_pod(U_final, S_final)
    X_final = U_final[:, PRESSURE_INPUT_COLUMNS]
    X_test = U_test[:, PRESSURE_INPUT_COLUMNS]
    C_test, S_test_projected = project_fields(S_test, pod)
    truncation = field_metrics(S_test_projected, S_test)

    # Shared basis checks: every regression method below receives this exact POD object.
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    test_case_rows: list[dict[str, Any]] = []

    for family, choice in selected.items():
        config = json.loads(choice["candidate_id"])
        predictor, fit_seconds, metadata = fit_candidate(
            family, config, X_final, pod["coefficients"], FINAL_TRAIN_SEED + len(rows) + 1
        )
        # Mean prediction time includes coefficient regression and POD
        # reconstruction. GP uncertainty estimation is measured separately.
        start_pred = time.perf_counter()
        C_pred = predictor(X_test)
        S_pred = reconstruct_fields(C_pred, pod)
        predict_seconds = time.perf_counter() - start_pred
        predictions[family] = S_pred

        C_std = None
        uncertainty_predict_seconds: float | str = ""
        if family == "GPR":
            uncertainty_started = time.perf_counter()
            _, C_std = predictor(X_test, return_std=True)
            uncertainty_predict_seconds = time.perf_counter() - uncertainty_started

        total = field_metrics(S_pred, S_test)
        regression_rel = relative_l2(S_pred, S_test_projected)
        total_sq = float(np.linalg.norm(S_test - S_pred) ** 2)
        trunc_sq = float(np.linalg.norm(S_test - S_test_projected) ** 2)
        regression_sq = float(np.linalg.norm(S_test_projected - S_pred) ** 2)
        decomposition_identity_error = abs(total_sq - trunc_sq - regression_sq) / max(
            total_sq, np.finfo(float).tiny
        )
        flux_started = time.perf_counter()
        q_pred = predict_fluxes(S_pred, U_test, field_shape)
        flux_postprocess_seconds = time.perf_counter() - flux_started
        q_scores = flux_metrics(q_pred, Q_test)

        coverage90 = coverage95 = width90 = width95 = ""
        if C_std is not None:
            z90, z95 = 1.6448536269514722, 1.959963984540054
            abs_resid = np.abs(C_test - C_pred)
            safe_std = np.maximum(C_std, 1e-14)
            coverage90 = float(np.mean(abs_resid <= z90 * safe_std))
            coverage95 = float(np.mean(abs_resid <= z95 * safe_std))
            width90 = float(np.mean(2.0 * z90 * C_std))
            width95 = float(np.mean(2.0 * z95 * C_std))

        row = {
            "model_family": family,
            "selected_config": choice["candidate_id"],
            "n_train": FINAL_N,
            "n_test": len(U_test),
            "pod_rank": pod["rank"],
            "pod_captured_energy": pod["captured_energy"],
            "pod_fit_seconds": pod["pod_fit_seconds"],
            "regressor_fit_seconds": fit_seconds,
            "prediction_seconds_batch": predict_seconds,
            "prediction_ms_per_case_including_POD_reconstruction": 1000.0 * predict_seconds / len(U_test),
            "uncertainty_prediction_seconds_batch": uncertainty_predict_seconds,
            "flux_postprocess_seconds_batch": flux_postprocess_seconds,
            "total_pressure_plus_flux_ms_per_case": (
                1000.0 * (predict_seconds + flux_postprocess_seconds) / len(U_test)
            ),
            "validation_rel_l2_mean_for_selection": choice["validation_rel_l2_mean"],
            "pod_truncation_rel_l2": truncation["relative_l2_global"],
            "regression_rel_l2_vs_projection": regression_rel,
            "error_decomposition_identity_relative_error": decomposition_identity_error,
            **total,
            **q_scores,
            "coefficient_coverage90": coverage90,
            "coefficient_coverage95": coverage95,
            "coefficient_mean_width90": width90,
            "coefficient_mean_width95": width95,
            **metadata,
        }
        rows.append(row)

        case_rel = np.linalg.norm(S_pred - S_test, axis=1) / np.maximum(
            np.linalg.norm(S_test, axis=1), np.finfo(float).tiny
        )
        flux_rel = np.abs(q_pred - Q_test) / np.maximum(np.abs(Q_test), np.finfo(float).tiny)
        for i in range(len(U_test)):
            physical = to_physical(U_test[i])
            test_case_rows.append({
                "model_family": family,
                "case_index": i,
                "logk_mean": physical[0],
                "logk_amp": physical[1],
                "pattern_angle_deg": physical[2],
                "field_relative_l2": case_rel[i],
                "q_left_true": Q_test[i],
                "q_left_pred": q_pred[i],
                "flux_relative_abs_error": flux_rel[i],
            })

        print(
            f"[final test] {family:22s} field rel-L2={total['relative_l2_global']:.5e} "
            f"flux rel-L2={q_scores['flux_relative_l2']:.5e}"
        )

    final_data = {
        "U_final": U_final,
        "S_final": S_final,
        "Q_final": Q_final,
        "pod": pod,
        "S_test_projected": S_test_projected,
        "test_case_rows": test_case_rows,
    }
    return rows, predictions, final_data


# =============================================================================
# 7. PLOTS AND SAVED OUTPUTS
# =============================================================================


def plot_validation_curves(summary: list[dict[str, Any]], out_path: Path) -> None:
    families = sorted({r["model_family"] for r in summary})
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for family in families:
        rows = sorted(
            [r for r in summary if r["model_family"] == family],
            key=lambda r: int(r["n_train"]),
        )
        if not rows:
            continue
        x = [int(r["n_train"]) for r in rows]
        y = [float(r["validation_rel_l2_mean"]) for r in rows]
        yerr = [float(r["validation_rel_l2_sd"]) for r in rows]
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.4, label=family)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(SAMPLE_SIZES, labels=[str(n) for n in SAMPLE_SIZES])
    ax.set_xlabel("number of full-order training snapshots")
    ax.set_ylabel("validation global relative L2 error")
    ax.set_title("Regression comparison on a shared POD basis")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_errors(
    S_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    field_shape: tuple[int, int],
    out_path: Path,
) -> None:
    families = list(predictions)
    ncols = 2
    nrows = int(math.ceil(len(families) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows), squeeze=False)
    for ax, family in zip(axes.ravel(), families):
        error = predictions[family] - S_test
        mean_abs = np.mean(np.abs(error), axis=0).reshape(field_shape)
        im = ax.imshow(mean_abs, origin="lower", aspect="auto")
        ax.set_title(f"{family}: mean absolute pressure error")
        ax.set_xlabel("x cell index")
        ax.set_ylabel("y cell index")
        fig.colorbar(im, ax=ax, shrink=0.82)
    for ax in axes.ravel()[len(families):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_test_parity(
    S_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    families = list(predictions)
    ncols = 2
    nrows = int(math.ceil(len(families) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows), squeeze=False)
    stride = max(1, S_test.size // 5000)
    truth = S_test.ravel()[::stride]
    for ax, family in zip(axes.ravel(), families):
        pred = predictions[family].ravel()[::stride]
        ax.scatter(truth, pred, s=3, alpha=0.25)
        lo = float(min(np.min(truth), np.min(pred)))
        hi = float(max(np.max(truth), np.max(pred)))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(family)
        ax.set_xlabel("full-order pressure")
        ax.set_ylabel("surrogate pressure")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(families):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_run_summary(
    summary_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "EXPERIMENT A — POD REGRESSION COMPARISON",
        "========================================",
        "",
        f"Parameter bounds: {dict(zip(PARAMETER_NAMES, zip(LOW.tolist(), HIGH.tolist())))}",
        f"Pressure input features: {PRESSURE_INPUT_NAMES}",
        f"Mesh: {NX} x {NY}; POD energy tolerance: {POD_TOL}",
        f"Training sizes: {SAMPLE_SIZES}; repeats: {N_REPEATS}; final common N: {FINAL_N}",
        "",
        "Validation-selected candidate at each N:",
    ]
    for row in sorted(summary_rows, key=lambda r: (r["model_family"], r["n_train"])):
        lines.append(
            f"  {row['model_family']:22s} N={int(row['n_train']):4d} "
            f"val={row['validation_rel_l2_mean']:.6e} "
            f"SD={row['validation_rel_l2_sd']:.3e} "
            f"config={row['candidate_id']}"
        )
    lines += ["", f"Final held-out test results (all models share N={FINAL_N} and one POD basis):"]
    for row in sorted(final_rows, key=lambda r: r["relative_l2_global"]):
        lines.append(
            f"  {row['model_family']:22s} field rel-L2={row['relative_l2_global']:.6e}; "
            f"regression-only rel-L2={row['regression_rel_l2_vs_projection']:.6e}; "
            f"POD truncation rel-L2={row['pod_truncation_rel_l2']:.6e}; "
            f"decomposition-check={row['error_decomposition_identity_relative_error']:.2e}; "
            f"flux rel-L2={row['flux_relative_l2']:.6e}; "
            f"fit={row['regressor_fit_seconds']:.3f}s; "
            f"predict={row['prediction_ms_per_case_including_POD_reconstruction']:.3f}ms/case; "
            f"flux-post={row['flux_postprocess_seconds_batch']:.3f}s"
        )
    lines += [
        "",
        "Interpretation notes:",
        "  - The POD basis is fitted from training snapshots only.",
        "  - Regressors within a training run receive the exact same POD basis and coefficients.",
        "  - Validation data select hyperparameters; the final test data are not used for selection.",
        "  - The final test comparison uses one common training set and one common POD basis.",
        "  - Pressure is structurally invariant to logk_mean for the supplied PDE and BCs.",
        "  - The pressure-active inputs are sampled jointly by a 2D LHS; logk_mean is sampled independently for flux.",
        "  - The 0--180 degree angle interval is treated as bounded and non-periodic; endpoint fields are checked.",
        "  - Flux uses all three parameters; predicted flux is post-processed from predicted pressure.",
        "  - B-splines are tensor-product cubic B-spline regression on scattered data, not exact interpolation.",
        "  - ThinPlateSpline is an RBFInterpolator thin-plate spline; it is distinct from the tensor B-spline model.",
        "  - GP interval coverage is measured in POD-coefficient space, not pointwise field space.",
        "  - These results are conditional on the solver, mesh, parameter bounds and boundary conditions.",
        "  - This experiment does not establish mesh independence; use your separate solver verification.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 8. MAIN
# =============================================================================


def main() -> None:
    validate_configuration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "LOW": LOW.tolist(),
        "HIGH": HIGH.tolist(),
        "parameter_names": PARAMETER_NAMES,
        "pressure_input_columns": list(PRESSURE_INPUT_COLUMNS),
        "pressure_input_names": PRESSURE_INPUT_NAMES,
        "angle_domain_interpretation": "bounded_nonperiodic_interval",
        "NX": NX,
        "NY": NY,
        "P_LEFT": P_LEFT,
        "P_RIGHT": P_RIGHT,
        "POD_TOL": POD_TOL,
        "SAMPLE_SIZES": SAMPLE_SIZES,
        "N_REPEATS": N_REPEATS,
        "FINAL_N": FINAL_N,
        "N_VALIDATION": N_VALIDATION,
        "N_TEST": N_TEST,
        "GPR_ALPHA": GPR_ALPHA,
        "GPR_RESTARTS": GPR_RESTARTS,
        "ANN_MAX_ITER": ANN_MAX_ITER,
        "MASTER_SEED": MASTER_SEED,
        "VALIDATION_SEED": VALIDATION_SEED,
        "TEST_SEED": TEST_SEED,
        "FINAL_TRAIN_SEED": FINAL_TRAIN_SEED,
    }
    (OUTPUT_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("=" * 78)
    print("EXPERIMENT A — REGRESSION COMPARISON ON A SHARED POD BASIS")
    print("=" * 78)
    print(f"Parameter bounds: {dict(zip(PARAMETER_NAMES, zip(LOW, HIGH)))}")
    print(f"Pressure regressors use: {PRESSURE_INPUT_NAMES}")
    print(f"Grid: {NX} x {NY}; N values: {SAMPLE_SIZES}; repeats: {N_REPEATS}")
    print(f"Validation cases: {N_VALIDATION}; final test cases: {N_TEST}; final common N: {FINAL_N}")
    if PRESSURE_INPUT_COLUMNS == (1, 2):
        print("NOTE: logk_mean is excluded from pressure-regression inputs because it is a global k scale.")
    print("NOTE: the angle range is treated as bounded and non-periodic; 0 and 180 degrees are distinct.")

    print("\n[setup] checking parameter invariances and angle endpoints")
    parameter_checks = verify_parameter_structure()
    (OUTPUT_DIR / "solver_parameter_checks.json").write_text(
        json.dumps(parameter_checks, indent=2), encoding="utf-8"
    )
    for key, value in parameter_checks.items():
        print(f"  {key}: {value:.6e}")

    # Independent validation and final test data; neither contributes to POD fitting.
    print("\n[setup] generating independent validation and final test sets")
    U_validation = make_lhs(N_VALIDATION, VALIDATION_SEED)
    S_validation, Q_validation, field_shape = solve_many(U_validation)
    U_test = make_lhs(N_TEST, TEST_SEED)
    S_test, Q_test, _ = solve_many(U_test)

    # Hyperparameter/model comparison using validation data only.
    started = time.perf_counter()
    validation_records = run_validation_experiment(U_validation, S_validation)
    validation_summary = summarize_validation(validation_records)
    write_csv(OUTPUT_DIR / "validation_records.csv", validation_records)
    write_csv(OUTPUT_DIR / "validation_best_by_family_and_N.csv", validation_summary)
    plot_validation_curves(validation_summary, OUTPUT_DIR / "validation_error_vs_samples.png")
    selected = select_final_candidates(validation_summary)

    expected_families = {family for family, _ in build_candidates()}
    missing_families = sorted(expected_families - set(selected))
    if missing_families:
        raise RuntimeError(
            "At least one regression family has no candidate successful in every "
            f"validation repetition at FINAL_N: {missing_families}. Inspect validation_records.csv."
        )

    print("\n[selection] validation-selected candidate for the common final training size")
    for family, row in selected.items():
        print(
            f"  {family:22s} config={row['candidate_id']} "
            f"validation={row['validation_rel_l2_mean']:.5e}"
        )

    # All model families now use one fresh final training design and one POD basis.
    final_rows, predictions, final_data = run_final_test(
        selected, U_test, S_test, Q_test, field_shape
    )
    write_csv(OUTPUT_DIR / "final_heldout_test_results.csv", final_rows)
    write_csv(OUTPUT_DIR / "final_test_case_errors.csv", final_data["test_case_rows"])
    plot_final_errors(S_test, predictions, field_shape, OUTPUT_DIR / "final_spatial_error_maps.png")
    plot_test_parity(S_test, predictions, OUTPUT_DIR / "final_test_parity.png")
    save_run_summary(validation_summary, final_rows, OUTPUT_DIR / "summary.txt")

    save_arrays: dict[str, Any] = {
        "U_validation": U_validation,
        "S_validation": S_validation,
        "Q_validation": Q_validation,
        "U_test": U_test,
        "S_test": S_test,
        "Q_test": Q_test,
        "U_final": final_data["U_final"],
        "S_final": final_data["S_final"],
        "Q_final": final_data["Q_final"],
        "pod_mean_field": final_data["pod"]["mean_field"],
        "pod_phi": final_data["pod"]["phi"],
        "S_test_projected": final_data["S_test_projected"],
        "validation_records_json": np.asarray(json.dumps(validation_records)),
        "validation_summary_json": np.asarray(json.dumps(validation_summary)),
        "final_test_rows_json": np.asarray(json.dumps(final_rows)),
        "field_shape": np.asarray(field_shape),
        "solver_parameter_checks_json": np.asarray(json.dumps(parameter_checks)),
    }
    for family, pred in predictions.items():
        safe_name = family.replace("-", "_").replace(".", "p")
        save_arrays[f"prediction_{safe_name}"] = pred
    np.savez_compressed(OUTPUT_DIR / "experiment_a_data.npz", **save_arrays)

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print("FINAL HELD-OUT TEST RESULTS")
    print("=" * 78)
    for row in sorted(final_rows, key=lambda r: r["relative_l2_global"]):
        print(
            f"{row['model_family']:22s} | field rel-L2={row['relative_l2_global']:.6e} "
            f"| flux rel-L2={row['flux_relative_l2']:.6e} "
            f"| fit={row['regressor_fit_seconds']:.2f}s "
            f"| predict={row['prediction_ms_per_case_including_POD_reconstruction']:.3f}ms/case"
        )
    print(f"\nValidation + final modelling elapsed: {elapsed / 60:.1f} min")
    print(f"All results saved in: {OUTPUT_DIR.resolve()}")
    print("Read summary.txt and final_heldout_test_results.csv before interpreting the ranking.")


if __name__ == "__main__":
    main()
