"""Reproducible comparison of sampling strategies and surrogate models.

Study design
------------
1. Generate independent validation and final test sets with the full Darcy solver.
2. Compare random, Latin-hypercube, Sobol, and uncertainty-driven adaptive sampling.
3. Compare POD + GPR against a direct multi-output neural network.
4. Repeat the training experiment for several independent seeds.
5. Select one configuration per model family using validation error only.
6. Train fresh final models and evaluate them once on the untouched test set.
7. Run Sobol sensitivity analysis on the final POD + GPR surrogate.

Expected project layout
-----------------------
    project_folder/
        corrected_surrogate_study.py
        darcy.py

The solver must expose:
    p, ..., ..., ... = solve_darcy(params=(logk_mean, logk_amp, angle_deg))

The returned pressure field may be any fixed-size array; it is flattened here.

IMPORTANT
---------
This script improves the experimental design and implementation. It does not
validate the mathematical correctness of darcy.py, its mesh, constitutive law,
or boundary conditions. Those still require a separate solver audit.
"""

from __future__ import annotations

import csv
import json
import math
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze
except ImportError as exc:
    raise ImportError(
        "SALib is required for the Sobol sensitivity analysis. Install it with: "
        "python -m pip install SALib"
    ) from exc

from darcy import solve_darcy


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

# These bounds and names are copied from the supplied Week 1 script.
# Confirm that they are physically appropriate for your Darcy model.
LOW = np.array([-1.0, 0.2, 0.0], dtype=float)
HIGH = np.array([1.0, 1.5, 180.0], dtype=float)
PARAMETER_NAMES = ["logk_mean", "logk_amp", "logk_angle"]
D = len(LOW)

# POD truncation: retain enough modes to capture at least 1 - POD_TOL
# of the snapshot fluctuation energy.
POD_TOL = 1e-4

# Training sizes are powers of two for a balanced Sobol design.
SAMPLE_SIZES = [16, 32, 64, 128, 256]
N_MAX = max(SAMPLE_SIZES)

# Three repetitions are a useful initial study. For stronger final evidence,
# increase this to 5-10 if the full-order solver is affordable.
N_REPEATS = 3

# Independent full-order evaluation sets. These are generated only once.
N_VALIDATION = 128
N_TEST = 256

# Adaptive design settings.
ADAPTIVE_INITIAL_N = min(SAMPLE_SIZES)
ADAPTIVE_BATCH_SIZE = 8
ADAPTIVE_CANDIDATE_POOL = 2000
ADAPTIVE_MIN_DISTANCE = 0.015  # Euclidean distance in normalized [0, 1]^D space
ADAPTIVE_FIXED_LENGTH_SCALE = 0.3

# GPR settings for the final/training comparisons.
GPR_RESTARTS = 1
GPR_ALPHA = 1e-8
GPR_LENGTH_SCALE_BOUNDS = (1e-2, 1e6)

# Direct NN baseline settings.
NN_HIDDEN_LAYERS = (256, 256)
NN_MAX_ITER = 3000
NN_ALPHA = 1e-4
NN_LEARNING_RATE = 1e-3
NN_EARLY_STOPPING = True

# Sobol sensitivity analysis settings.
SOBOL_BASE_N = 4096
SOBOL_CONVERGENCE_N = [256, 512, 1024, 2048, 4096]
SOBOL_SEED = 4107
SOBOL_BOOTSTRAP_SEED = 4108
SOBOL_NUM_RESAMPLES = 200

MASTER_SEED = 20261110
VALIDATION_SEED = 11001
TEST_SEED = 22002
FINAL_TRAIN_SEED = 33003

OUTPUT_DIR = Path("corrected_surrogate_results")


# =============================================================================
# 2. VALIDATION AND SMALL UTILITIES
# =============================================================================


def validate_configuration() -> None:
    if LOW.shape != HIGH.shape or LOW.ndim != 1:
        raise ValueError("LOW and HIGH must be one-dimensional arrays of equal length.")
    if len(LOW) != D:
        raise ValueError("Parameter dimension D does not match LOW/HIGH.")
    if np.any(HIGH <= LOW):
        raise ValueError("Every upper parameter bound must exceed its lower bound.")
    if not (0.0 < POD_TOL < 1.0):
        raise ValueError("POD_TOL must lie between 0 and 1.")
    if sorted(SAMPLE_SIZES) != SAMPLE_SIZES:
        raise ValueError("SAMPLE_SIZES must be sorted in ascending order.")
    if SAMPLE_SIZES[0] < 2:
        raise ValueError("The smallest training size must be at least 2.")
    if any(n & (n - 1) for n in SAMPLE_SIZES):
        raise ValueError("All SAMPLE_SIZES must be powers of two for this Sobol design.")
    if any(n & (n - 1) for n in SOBOL_CONVERGENCE_N):
        raise ValueError("All SOBOL_CONVERGENCE_N values must be powers of two.")
    if SOBOL_BASE_N not in SOBOL_CONVERGENCE_N:
        raise ValueError("SOBOL_BASE_N must be included in SOBOL_CONVERGENCE_N.")
    if max(SOBOL_CONVERGENCE_N) > SOBOL_BASE_N:
        raise ValueError("SOBOL_CONVERGENCE_N cannot exceed SOBOL_BASE_N.")
    if ADAPTIVE_INITIAL_N > N_MAX:
        raise ValueError("ADAPTIVE_INITIAL_N cannot exceed N_MAX.")
    if N_REPEATS < 2:
        warnings.warn(
            "N_REPEATS < 2 gives no meaningful estimate of between-run variability.",
            RuntimeWarning,
        )


def to_params(U: np.ndarray) -> np.ndarray:
    """Map normalized inputs in [0, 1]^D to physical parameter values."""
    U = np.asarray(U, dtype=float)
    return LOW + U * (HIGH - LOW)


def make_lhs(n: int, seed: int) -> np.ndarray:
    return qmc.LatinHypercube(d=D, seed=seed).random(n)


def make_uniform_random(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((n, D))


def make_sobol_design(n: int, seed: int) -> np.ndarray:
    """Generate a scrambled Sobol design; n must be a power of two."""
    if n & (n - 1):
        raise ValueError(f"Sobol sample size must be a power of two; received {n}.")
    m = int(round(math.log2(n)))
    engine = qmc.Sobol(d=D, scramble=True, seed=seed)
    return engine.random_base2(m=m)


def solve_many(U: np.ndarray) -> np.ndarray:
    """Run the full-order model for each normalized parameter vector."""
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[1] != D:
        raise ValueError(f"U must have shape (n, {D}); received {U.shape}.")

    snapshots: List[np.ndarray] = []
    expected_shape = None
    for i, u in enumerate(U):
        physical_params = tuple(float(x) for x in to_params(u))
        result = solve_darcy(params=physical_params)
        if not isinstance(result, (tuple, list)) or len(result) < 1:
            raise TypeError(
                "solve_darcy must return a tuple/list whose first item is the pressure field."
            )
        pressure = np.asarray(result[0], dtype=float).ravel()
        if pressure.size == 0 or not np.all(np.isfinite(pressure)):
            raise ValueError(f"Non-finite or empty pressure field at sample {i}.")
        if expected_shape is None:
            expected_shape = pressure.shape
        elif pressure.shape != expected_shape:
            raise ValueError(
                "The solver returned pressure fields with inconsistent sizes: "
                f"expected {expected_shape}, received {pressure.shape} at sample {i}."
            )
        snapshots.append(pressure)

    return np.vstack(snapshots)


def error_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Return global and per-case relative L2 errors."""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction shape {pred.shape} != truth shape {truth.shape}.")

    tiny = np.finfo(float).tiny
    global_rel = float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), tiny))
    case_num = np.linalg.norm(pred - truth, axis=1)
    case_den = np.maximum(np.linalg.norm(truth, axis=1), tiny)
    case_rel = case_num / case_den
    return {
        "relative_l2_global": global_rel,
        "relative_l2_case_mean": float(np.mean(case_rel)),
        "relative_l2_case_median": float(np.median(case_rel)),
        "relative_l2_case_max": float(np.max(case_rel)),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# 3. POD + GPR SURROGATE
# =============================================================================


def fit_gpr(X: np.ndarray, y: np.ndarray, fixed_hp: bool = False):
    """Fit one scalar GPR model to one POD coefficient."""
    if fixed_hp:
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
            length_scale=ADAPTIVE_FIXED_LENGTH_SCALE,
            length_scale_bounds="fixed",
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=GPR_ALPHA,
            optimizer=None,
            normalize_y=True,
        )
    else:
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(X.shape[1]),
            length_scale_bounds=GPR_LENGTH_SCALE_BOUNDS,
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=GPR_ALPHA,
            n_restarts_optimizer=GPR_RESTARTS,
            normalize_y=True,
            random_state=MASTER_SEED,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def fit_pod_gpr(
    U: np.ndarray,
    S: np.ndarray,
    fixed_hp: bool = False,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], int]:
    """Fit POD basis + independent GPR models for the POD coefficients.

    Returns
    -------
    predict:
        Maps normalized parameters to reconstructed pressure fields.
    integrated_std:
        Approximate integrated posterior standard deviation in field space.
        Because the POD basis is orthonormal and coefficient GPs are treated
        as independent, this is sqrt(sum_k std_k**2).
    rank:
        Number of retained POD modes.
    """
    U = np.asarray(U, dtype=float)
    S = np.asarray(S, dtype=float)
    if U.ndim != 2 or U.shape[1] != D:
        raise ValueError(f"U must have shape (n, {D}).")
    if S.ndim != 2 or S.shape[0] != U.shape[0]:
        raise ValueError("S must be a 2D snapshot matrix with one row per input sample.")

    mean_field = S.mean(axis=0)
    centered = S - mean_field
    _, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
    total_energy = float(np.sum(singular_values ** 2))

    if total_energy <= np.finfo(float).eps:
        def predict_constant(Uq: np.ndarray) -> np.ndarray:
            return np.repeat(mean_field[None, :], len(Uq), axis=0)

        def zero_uncertainty(Uq: np.ndarray) -> np.ndarray:
            return np.zeros(len(Uq), dtype=float)

        return predict_constant, zero_uncertainty, 0

    cumulative_energy = np.cumsum(singular_values ** 2) / total_energy
    rank = int(np.searchsorted(cumulative_energy, 1.0 - POD_TOL) + 1)
    rank = max(1, min(rank, Vt.shape[0]))

    phi = Vt[:rank].T  # (number of field degrees of freedom, rank)
    coefficients = centered @ phi
    gps = [fit_gpr(U, coefficients[:, k], fixed_hp=fixed_hp) for k in range(rank)]

    def predict(Uq: np.ndarray) -> np.ndarray:
        Uq = np.asarray(Uq, dtype=float)
        coefficient_means = np.column_stack([gp.predict(Uq) for gp in gps])
        return mean_field + coefficient_means @ phi.T

    def integrated_std(Uq: np.ndarray) -> np.ndarray:
        Uq = np.asarray(Uq, dtype=float)
        coefficient_stds = np.column_stack(
            [gp.predict(Uq, return_std=True)[1] for gp in gps]
        )
        # Correct integrated variance for an orthonormal POD basis.
        # Do not project standard deviations and then square the projection:
        # that introduces spurious cross terms.
        return np.sqrt(np.sum(coefficient_stds ** 2, axis=1))

    return predict, integrated_std, rank


# =============================================================================
# 4. DIRECT NEURAL NETWORK BASELINE
# =============================================================================


def fit_direct_nn(U: np.ndarray, S: np.ndarray, seed: int) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a direct parameter-to-full-pressure-field MLP."""
    U = np.asarray(U, dtype=float)
    S = np.asarray(S, dtype=float)

    input_scaler = StandardScaler().fit(U)
    output_mean = S.mean(axis=0)
    output_std = S.std(axis=0)
    # Avoid division by almost-zero values for constant/near-constant DOFs.
    output_std = np.where(output_std < 1e-12, 1.0, output_std)
    S_scaled = (S - output_mean) / output_std

    nn = MLPRegressor(
        hidden_layer_sizes=NN_HIDDEN_LAYERS,
        activation="tanh",
        solver="adam",
        alpha=NN_ALPHA,
        learning_rate_init=NN_LEARNING_RATE,
        max_iter=NN_MAX_ITER,
        early_stopping=NN_EARLY_STOPPING,
        validation_fraction=0.15,
        n_iter_no_change=60,
        tol=1e-5,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        nn.fit(input_scaler.transform(U), S_scaled)

    def predict(Uq: np.ndarray) -> np.ndarray:
        Uq = np.asarray(Uq, dtype=float)
        return nn.predict(input_scaler.transform(Uq)) * output_std + output_mean

    return predict


# =============================================================================
# 5. SAMPLING DESIGNS
# =============================================================================


def make_design(strategy: str, n: int, seed: int) -> np.ndarray:
    if strategy == "random":
        return make_uniform_random(n, seed)
    if strategy == "LHS":
        return make_lhs(n, seed)
    if strategy == "Sobol":
        return make_sobol_design(n, seed)
    raise ValueError(f"Unknown non-adaptive sampling strategy: {strategy}")


def fit_adaptive_design(n_total: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build an uncertainty-driven adaptive design and retain its snapshots.

    The first ADAPTIVE_INITIAL_N points are an LHS design. The next points are
    selected in batches from a random candidate pool by maximizing the
    integrated POD-GPR posterior standard deviation. The acquisition model
    uses a fixed length scale to keep repeated refitting affordable.

    The batch is selected from one posterior update; it is therefore a
    batched, approximate adaptive strategy, not a globally optimal design.
    """
    if n_total < ADAPTIVE_INITIAL_N:
        raise ValueError("n_total is smaller than ADAPTIVE_INITIAL_N.")

    rng = np.random.default_rng(seed)
    U = make_lhs(ADAPTIVE_INITIAL_N, seed + 1)
    S = solve_many(U)

    while len(U) < n_total:
        _, uncertainty_fn, _ = fit_pod_gpr(U, S, fixed_hp=True)
        candidates = rng.random((ADAPTIVE_CANDIDATE_POOL, D))
        scores = uncertainty_fn(candidates)

        # Avoid choosing candidates that duplicate or nearly duplicate existing points.
        distance_to_existing = cdist(candidates, U).min(axis=1)
        scores[distance_to_existing < ADAPTIVE_MIN_DISTANCE] = -np.inf

        batch_count = min(ADAPTIVE_BATCH_SIZE, n_total - len(U))
        order = np.argsort(scores)[::-1]
        selected: List[np.ndarray] = []

        for idx in order:
            if not np.isfinite(scores[idx]):
                break
            candidate = candidates[idx]
            if selected:
                d_selected = np.linalg.norm(np.vstack(selected) - candidate, axis=1)
                if np.min(d_selected) < ADAPTIVE_MIN_DISTANCE:
                    continue
            selected.append(candidate)
            if len(selected) == batch_count:
                break

        # Defensive fallback in the unlikely event that filtering leaves too few candidates.
        if len(selected) < batch_count:
            for candidate in rng.random((batch_count - len(selected), D)):
                selected.append(candidate)

        U_new = np.vstack(selected)
        S_new = solve_many(U_new)
        U = np.vstack([U, U_new])
        S = np.vstack([S, S_new])

    return U, S


# =============================================================================
# 6. VALIDATION EXPERIMENT
# =============================================================================


def evaluate_one_configuration(
    strategy: str,
    U_train: np.ndarray,
    S_train: np.ndarray,
    U_validation: np.ndarray,
    S_validation: np.ndarray,
    repeat: int,
    n_train: int,
    seed: int,
) -> List[Dict]:
    rows: List[Dict] = []

    pod_predict, _, pod_rank = fit_pod_gpr(U_train, S_train, fixed_hp=False)
    pod_metrics = error_metrics(pod_predict(U_validation), S_validation)
    rows.append({
        "repeat": repeat,
        "strategy": strategy,
        "n_train": n_train,
        "model": "POD+GPR",
        "pod_rank": pod_rank,
        **pod_metrics,
    })

    nn_predict = fit_direct_nn(U_train, S_train, seed=seed)
    nn_metrics = error_metrics(nn_predict(U_validation), S_validation)
    rows.append({
        "repeat": repeat,
        "strategy": strategy,
        "n_train": n_train,
        "model": "direct_NN",
        "pod_rank": "",
        **nn_metrics,
    })
    return rows


def run_validation_experiment(
    U_validation: np.ndarray,
    S_validation: np.ndarray,
) -> List[Dict]:
    """Compare designs on validation data, never on the final test set."""
    records: List[Dict] = []
    strategies = ["random", "LHS", "Sobol", "adaptive"]

    expected_training_solves = 0
    started = time.time()

    for repeat in range(N_REPEATS):
        for strategy_idx, strategy in enumerate(strategies):
            seed = MASTER_SEED + repeat * 10000 + strategy_idx * 1000
            print(f"\n[validation] repeat={repeat + 1}/{N_REPEATS}, strategy={strategy}")

            if strategy == "adaptive":
                U_all, S_all = fit_adaptive_design(N_MAX, seed)
                expected_training_solves += N_MAX
                for n in SAMPLE_SIZES:
                    rows = evaluate_one_configuration(
                        strategy, U_all[:n], S_all[:n],
                        U_validation, S_validation, repeat, n, seed + n,
                    )
                    records.extend(rows)
                    print(
                        f"  N={n:3d} | POD+GPR={rows[0]['relative_l2_global']:.4e} "
                        f"| NN={rows[1]['relative_l2_global']:.4e}"
                    )

            elif strategy in ("random", "Sobol"):
                # One maximum design; prefixes give nested designs across budgets.
                U_all = make_design(strategy, N_MAX, seed)
                S_all = solve_many(U_all)
                expected_training_solves += N_MAX
                for n in SAMPLE_SIZES:
                    rows = evaluate_one_configuration(
                        strategy, U_all[:n], S_all[:n],
                        U_validation, S_validation, repeat, n, seed + n,
                    )
                    records.extend(rows)
                    print(
                        f"  N={n:3d} | POD+GPR={rows[0]['relative_l2_global']:.4e} "
                        f"| NN={rows[1]['relative_l2_global']:.4e}"
                    )

            elif strategy == "LHS":
                # A fresh LHS is generated at each size because a prefix of a
                # larger LHS is not generally an LHS design at the smaller size.
                for n in SAMPLE_SIZES:
                    U_train = make_lhs(n, seed + n)
                    S_train = solve_many(U_train)
                    expected_training_solves += n
                    rows = evaluate_one_configuration(
                        strategy, U_train, S_train,
                        U_validation, S_validation, repeat, n, seed + n + 1,
                    )
                    records.extend(rows)
                    print(
                        f"  N={n:3d} | POD+GPR={rows[0]['relative_l2_global']:.4e} "
                        f"| NN={rows[1]['relative_l2_global']:.4e}"
                    )

    elapsed = time.time() - started
    print(
        f"\nValidation experiment finished in {elapsed / 60:.1f} min. "
        f"Training full-order solves (counted from design generation): {expected_training_solves}."
    )
    return records


def summarize_validation(records: List[Dict]) -> List[Dict]:
    """Aggregate repetitions and return mean/std for each configuration."""
    keys = sorted({(r["model"], r["strategy"], int(r["n_train"])) for r in records})
    summary: List[Dict] = []
    for model, strategy, n_train in keys:
        selected = [
            r for r in records
            if r["model"] == model
            and r["strategy"] == strategy
            and int(r["n_train"]) == n_train
        ]
        values = np.array([r["relative_l2_global"] for r in selected], dtype=float)
        case_values = np.array([r["relative_l2_case_mean"] for r in selected], dtype=float)
        summary.append({
            "model": model,
            "strategy": strategy,
            "n_train": n_train,
            "n_repeats": len(selected),
            "validation_rel_l2_mean": float(np.mean(values)),
            "validation_rel_l2_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "validation_case_rel_l2_mean": float(np.mean(case_values)),
            "validation_case_rel_l2_std": (
                float(np.std(case_values, ddof=1)) if len(case_values) > 1 else 0.0
            ),
        })
    return summary


def select_best_configurations(summary: List[Dict]) -> Dict[str, Dict]:
    """Select one configuration per model family using validation means only."""
    best: Dict[str, Dict] = {}
    for model in sorted({row["model"] for row in summary}):
        candidates = [row for row in summary if row["model"] == model]
        # Tie-break toward fewer expensive full-order training solves.
        winner = min(
            candidates,
            key=lambda row: (row["validation_rel_l2_mean"], row["n_train"]),
        )
        best[model] = winner
    return best


# =============================================================================
# 7. FINAL TRAINING AND ONE-TIME TEST EVALUATION
# =============================================================================


def generate_final_training_data(strategy: str, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if strategy == "adaptive":
        return fit_adaptive_design(n, seed)
    U = make_design(strategy, n, seed)
    return U, solve_many(U)


def train_and_test_selected_models(
    best_configs: Dict[str, Dict],
    U_test: np.ndarray,
    S_test: np.ndarray,
) -> Tuple[List[Dict], Dict[str, np.ndarray], Dict[str, Dict]]:
    """Freshly train selected configurations; evaluate only on held-out test data."""
    final_test_rows: List[Dict] = []
    predictions: Dict[str, np.ndarray] = {}
    final_models: Dict[str, Dict] = {}

    for model_idx, model_name in enumerate(["POD+GPR", "direct_NN"]):
        config = best_configs[model_name]
        strategy = str(config["strategy"])
        n_train = int(config["n_train"])
        seed = FINAL_TRAIN_SEED + model_idx * 10000

        print(
            f"\n[final] training {model_name}: strategy={strategy}, N={n_train} "
            "(selected using validation data)"
        )
        U_train, S_train = generate_final_training_data(strategy, n_train, seed)

        if model_name == "POD+GPR":
            predictor, _, pod_rank = fit_pod_gpr(U_train, S_train, fixed_hp=False)
            pred_test = predictor(U_test)
            final_models[model_name] = {
                "U_train": U_train,
                "S_train": S_train,
                "predict": predictor,
                "pod_rank": pod_rank,
                "strategy": strategy,
                "n_train": n_train,
            }
        else:
            predictor = fit_direct_nn(U_train, S_train, seed=seed)
            pred_test = predictor(U_test)
            final_models[model_name] = {
                "U_train": U_train,
                "S_train": S_train,
                "predict": predictor,
                "strategy": strategy,
                "n_train": n_train,
            }

        metrics = error_metrics(pred_test, S_test)
        predictions[model_name] = pred_test
        final_test_rows.append({
            "model": model_name,
            "strategy_selected_on_validation": strategy,
            "n_train_selected_on_validation": n_train,
            "validation_rel_l2_mean": float(config["validation_rel_l2_mean"]),
            "test_set_size": len(U_test),
            "pod_rank": final_models[model_name].get("pod_rank", ""),
            **metrics,
        })
        print(f"  untouched test relative L2 error: {metrics['relative_l2_global']:.6e}")

    return final_test_rows, predictions, final_models


# =============================================================================
# 8. SOBOL GLOBAL SENSITIVITY ANALYSIS
# =============================================================================


def run_sobol_analysis(predictor: Callable[[np.ndarray], np.ndarray]) -> Tuple[Dict, Dict[str, List[float]]]:
    """Sobol analysis of domain-averaged pressure using the final POD+GPR model.

    The three inputs are assumed independent and uniformly distributed over
    LOW/HIGH. The scalar quantity of interest (QoI) is the spatial mean pressure.
    """
    problem = {
        "num_vars": D,
        "names": PARAMETER_NAMES,
        "bounds": [[float(LOW[i]), float(HIGH[i])] for i in range(D)],
    }

    convergence: Dict[str, List[float]] = {
        "N": [],
        "S1": [],
        "ST": [],
        "S1_conf": [],
        "ST_conf": [],
    }
    final_si = None

    for n_base in SOBOL_CONVERGENCE_N:
        if n_base > SOBOL_BASE_N:
            raise ValueError("SOBOL_CONVERGENCE_N cannot exceed SOBOL_BASE_N.")

        # Reusing the same seed makes the scrambled low-discrepancy sequences
        # reproducible and gives nested base-sequence prefixes across powers of two.
        X_sa = sobol_sample.sample(
            problem,
            n_base,
            calc_second_order=False,
            scramble=True,
            seed=SOBOL_SEED,
        )
        U_sa = (X_sa - LOW) / (HIGH - LOW)
        predicted_fields = predictor(U_sa)
        qoi = predicted_fields.mean(axis=1)  # domain-averaged pressure

        if not np.all(np.isfinite(qoi)):
            raise ValueError("Non-finite QoI values encountered during Sobol analysis.")
        if np.var(qoi) <= np.finfo(float).eps:
            raise ValueError(
                "The domain-averaged pressure has nearly zero variance; Sobol indices "
                "are not identifiable for this QoI over the selected parameter ranges."
            )

        si = sobol_analyze.analyze(
            problem,
            qoi,
            calc_second_order=False,
            num_resamples=SOBOL_NUM_RESAMPLES,
            conf_level=0.95,
            print_to_console=False,
            seed=SOBOL_BOOTSTRAP_SEED,
        )

        convergence["N"].append(int(n_base))
        convergence["S1"].append(np.asarray(si["S1"], dtype=float).tolist())
        convergence["ST"].append(np.asarray(si["ST"], dtype=float).tolist())
        convergence["S1_conf"].append(np.asarray(si["S1_conf"], dtype=float).tolist())
        convergence["ST_conf"].append(np.asarray(si["ST_conf"], dtype=float).tolist())
        final_si = si
        print(
            f"[Sobol] base N={n_base:4d}, model evaluations={len(qoi):6d}, "
            f"S1={np.array2string(np.asarray(si['S1']), precision=4)}"
        )

    if final_si is None or SOBOL_BASE_N not in convergence["N"]:
        raise ValueError("The final Sobol base sample size must be included in the convergence list.")

    return {k: np.asarray(v) if k != "N" else np.asarray(v, dtype=int)
            for k, v in final_si.items() if isinstance(v, (list, np.ndarray))}, convergence


# =============================================================================
# 9. PLOTS AND OUTPUTS
# =============================================================================


def plot_validation_curves(summary: List[Dict], out_path: Path) -> None:
    strategies = ["random", "LHS", "Sobol", "adaptive"]
    styles = {
        "random": ("o-", "C0"),
        "LHS": ("s-", "C1"),
        "Sobol": ("^-", "C2"),
        "adaptive": ("D-", "C3"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, ["POD+GPR", "direct_NN"]):
        for strategy in strategies:
            rows = sorted(
                [r for r in summary if r["model"] == model and r["strategy"] == strategy],
                key=lambda r: r["n_train"],
            )
            if not rows:
                continue
            x = np.array([r["n_train"] for r in rows])
            y = np.array([r["validation_rel_l2_mean"] for r in rows])
            yerr = np.array([r["validation_rel_l2_std"] for r in rows])
            # Keep the lower error bar positive on the logarithmic y-axis.
            yerr_lower = np.minimum(yerr, 0.99 * y)
            yerr_plot = np.vstack([yerr_lower, yerr])
            ax.errorbar(
                x, y, yerr=yerr_plot, fmt=styles[strategy][0], color=styles[strategy][1],
                capsize=3, linewidth=1.5, markersize=5, label=strategy,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(SAMPLE_SIZES, labels=[str(n) for n in SAMPLE_SIZES])
        ax.set_xlabel("number of full-order training solves")
        ax.set_title(model)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("validation relative L2 error (mean ± SD)")
    fig.suptitle(f"Sampling strategy × surrogate model ({N_REPEATS} independent runs)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_parity(
    S_test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    final_test_rows: List[Dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, model_name in zip(axes, ["POD+GPR", "direct_NN"]):
        pred = predictions[model_name]
        # Deterministic subsampling avoids plotting every field value for large meshes.
        x = S_test.ravel()[::8]
        y = pred.ravel()[::8]
        ax.scatter(x, y, s=4, alpha=0.3, edgecolors="none")
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        row = next(r for r in final_test_rows if r["model"] == model_name)
        ax.set_xlabel("full-order pressure")
        ax.set_title(
            f"{model_name}\nN={row['n_train_selected_on_validation']}, "
            f"test rel. L2={row['relative_l2_global']:.2e}"
        )
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("surrogate pressure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sobol_indices(si: Dict, out_path: Path) -> None:
    s1 = np.asarray(si["S1"], dtype=float)
    st = np.asarray(si["ST"], dtype=float)
    s1_conf = np.asarray(si["S1_conf"], dtype=float)
    st_conf = np.asarray(si["ST_conf"], dtype=float)
    x = np.arange(D)
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, s1, width, yerr=s1_conf, capsize=3, label="first-order")
    ax.bar(x + width / 2, st, width, yerr=st_conf, capsize=3, label="total-order")
    ax.set_xticks(x, PARAMETER_NAMES)
    ax.set_ylabel("Sobol index (domain-averaged pressure)")
    ax.set_title(f"Sobol sensitivity ({SOBOL_BASE_N} base samples)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sobol_convergence(convergence: Dict[str, List[float]], out_path: Path) -> None:
    n_values = np.asarray(convergence["N"], dtype=int)
    s1 = np.asarray(convergence["S1"], dtype=float)  # (n_sizes, D)
    s1_conf = np.asarray(convergence["S1_conf"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, name in enumerate(PARAMETER_NAMES):
        ax.errorbar(
            n_values, s1[:, j], yerr=s1_conf[:, j], marker="o", capsize=3,
            linewidth=1.4, label=name,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values, labels=[str(n) for n in n_values])
    ax.set_xlabel("Sobol base sample size")
    ax.set_ylabel("first-order Sobol index (± 95% bootstrap interval)")
    ax.set_title("Sobol estimate convergence on the final surrogate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_worst_case_error(
    S_test: np.ndarray,
    pod_prediction: np.ndarray,
    out_path: Path,
    error_csv: Path,
) -> None:
    errors = pod_prediction - S_test
    per_case_max = np.max(np.abs(errors), axis=1)
    worst_idx = int(np.argmax(per_case_max))
    error_field = errors[worst_idx]

    # Plot a spatial map only when the returned field is a square grid.
    n_dof = error_field.size
    side = int(round(math.sqrt(n_dof)))
    if side * side == n_dof:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(error_field.reshape(side, side), origin="lower", cmap="RdBu_r")
        ax.set_xlabel("grid x-index")
        ax.set_ylabel("grid y-index")
        ax.set_title(f"POD+GPR pressure error: worst test case {worst_idx}")
        fig.colorbar(im, ax=ax, label="surrogate − full-order pressure")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        print(
            f"[plot] Skipping spatial error map: {n_dof} DOFs is not a square grid. "
            "The solver's grid dimensions are not available in this script."
        )

    np.savetxt(error_csv, errors, delimiter=",")


def save_summary(
    best_configs: Dict[str, Dict],
    final_test_rows: List[Dict],
    si: Dict,
    output_path: Path,
) -> None:
    lines = [
        "CORRECTED SURROGATE STUDY SUMMARY",
        "=================================",
        "",
        "Validation-selected configurations (selection did not use the final test set):",
    ]
    for model, row in best_configs.items():
        lines.append(
            f"  {model}: strategy={row['strategy']}, N={row['n_train']}, "
            f"validation mean={row['validation_rel_l2_mean']:.6e}, "
            f"validation SD={row['validation_rel_l2_std']:.6e}"
        )

    lines += ["", "Final held-out test results:"]
    for row in final_test_rows:
        lines.append(
            f"  {row['model']}: strategy={row['strategy_selected_on_validation']}, "
            f"N={row['n_train_selected_on_validation']}, "
            f"test global rel. L2={row['relative_l2_global']:.6e}, "
            f"mean case rel. L2={row['relative_l2_case_mean']:.6e}, "
            f"max case rel. L2={row['relative_l2_case_max']:.6e}"
        )

    lines += ["", "Sobol indices for domain-averaged pressure:"]
    for name, s1, st, s1c, stc in zip(
        PARAMETER_NAMES, si["S1"], si["ST"], si["S1_conf"], si["ST_conf"]
    ):
        lines.append(
            f"  {name}: S1={s1:.6f} ± {s1c:.6f}; "
            f"ST={st:.6f} ± {stc:.6f}"
        )

    lines += [
        "",
        "Interpretation limits:",
        "  - Results are conditional on the Darcy solver, parameter bounds, and chosen QoI.",
        "  - Sobol analysis assumes independent uniform inputs over the stated bounds.",
        "  - The Sobol QoI is the mean of the predicted pressure field, not flow rate.",
        "  - Sobol bootstrap intervals do not include surrogate-training or model-form uncertainty.",
        "  - The adaptive design uses a fixed-hyperparameter GP acquisition model and batches.",
        "  - This script does not independently verify the mathematical correctness of darcy.py.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 10. MAIN
# =============================================================================


def main() -> None:
    validate_configuration()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "LOW": LOW.tolist(),
        "HIGH": HIGH.tolist(),
        "parameter_names": PARAMETER_NAMES,
        "POD_TOL": POD_TOL,
        "SAMPLE_SIZES": SAMPLE_SIZES,
        "N_REPEATS": N_REPEATS,
        "N_VALIDATION": N_VALIDATION,
        "N_TEST": N_TEST,
        "adaptive_initial_n": ADAPTIVE_INITIAL_N,
        "adaptive_batch_size": ADAPTIVE_BATCH_SIZE,
        "adaptive_candidate_pool": ADAPTIVE_CANDIDATE_POOL,
        "adaptive_min_distance": ADAPTIVE_MIN_DISTANCE,
        "GPR_RESTARTS": GPR_RESTARTS,
        "NN_HIDDEN_LAYERS": list(NN_HIDDEN_LAYERS),
        "SOBOL_BASE_N": SOBOL_BASE_N,
        "SOBOL_CONVERGENCE_N": SOBOL_CONVERGENCE_N,
        "MASTER_SEED": MASTER_SEED,
        "VALIDATION_SEED": VALIDATION_SEED,
        "TEST_SEED": TEST_SEED,
        "FINAL_TRAIN_SEED": FINAL_TRAIN_SEED,
    }
    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("CORRECTED SURROGATE MODELLING EXPERIMENT")
    print("=" * 72)
    print(f"Parameter bounds: {dict(zip(PARAMETER_NAMES, zip(LOW, HIGH)))}")
    print(f"Training sizes: {SAMPLE_SIZES}; repetitions: {N_REPEATS}")
    print(f"Validation cases: {N_VALIDATION}; final test cases: {N_TEST}")

    # These two sets are independent of all training designs.
    print("\n[setup] generating independent validation and final test data")
    U_validation = make_lhs(N_VALIDATION, VALIDATION_SEED)
    S_validation = solve_many(U_validation)
    U_test = make_lhs(N_TEST, TEST_SEED)
    S_test = solve_many(U_test)

    # Stage 1: validation-only comparison and configuration selection.
    validation_records = run_validation_experiment(U_validation, S_validation)
    validation_summary = summarize_validation(validation_records)
    best_configs = select_best_configurations(validation_summary)

    write_csv(OUTPUT_DIR / "validation_results_by_run.csv", validation_records)
    write_csv(OUTPUT_DIR / "validation_summary.csv", validation_summary)
    plot_validation_curves(validation_summary, OUTPUT_DIR / "validation_error_vs_samples.png")

    print("\n[selection] best configurations selected using validation data only")
    for model, row in best_configs.items():
        print(
            f"  {model}: {row['strategy']} / N={row['n_train']} / "
            f"validation mean={row['validation_rel_l2_mean']:.5e} "
            f"± {row['validation_rel_l2_std']:.2e}"
        )

    # Stage 2: fresh final training data; test data has not influenced selection.
    final_test_rows, predictions, final_models = train_and_test_selected_models(
        best_configs, U_test, S_test
    )
    write_csv(OUTPUT_DIR / "final_heldout_test_results.csv", final_test_rows)
    plot_final_parity(
        S_test, predictions, final_test_rows,
        OUTPUT_DIR / "final_test_parity.png",
    )

    # Stage 3: sensitivity analysis using the final POD+GPR model.
    print("\n[sensitivity] Sobol analysis on final POD+GPR surrogate")
    sobol_result, convergence = run_sobol_analysis(final_models["POD+GPR"]["predict"])
    write_csv(
        OUTPUT_DIR / "sobol_indices.csv",
        [
            {
                "parameter": name,
                "S1": float(sobol_result["S1"][i]),
                "S1_conf": float(sobol_result["S1_conf"][i]),
                "ST": float(sobol_result["ST"][i]),
                "ST_conf": float(sobol_result["ST_conf"][i]),
            }
            for i, name in enumerate(PARAMETER_NAMES)
        ],
    )
    plot_sobol_indices(sobol_result, OUTPUT_DIR / "sobol_indices.png")
    plot_sobol_convergence(convergence, OUTPUT_DIR / "sobol_convergence.png")

    plot_worst_case_error(
        S_test,
        predictions["POD+GPR"],
        OUTPUT_DIR / "worst_case_pressure_error.png",
        OUTPUT_DIR / "test_pressure_errors.csv",
    )

    save_summary(
        best_configs,
        final_test_rows,
        sobol_result,
        OUTPUT_DIR / "summary.txt",
    )

    # Save the data needed to reproduce figures and inspect predictions later.
    np.savez_compressed(
        OUTPUT_DIR / "corrected_study_data.npz",
        U_validation=U_validation,
        S_validation=S_validation,
        U_test=U_test,
        S_test=S_test,
        pod_test_prediction=predictions["POD+GPR"],
        nn_test_prediction=predictions["direct_NN"],
        final_pod_U_train=final_models["POD+GPR"]["U_train"],
        final_pod_S_train=final_models["POD+GPR"]["S_train"],
        final_nn_U_train=final_models["direct_NN"]["U_train"],
        final_nn_S_train=final_models["direct_NN"]["S_train"],
        validation_records_json=np.asarray(json.dumps(validation_records)),
        validation_summary_json=np.asarray(json.dumps(validation_summary)),
        final_test_rows_json=np.asarray(json.dumps(final_test_rows)),
        sobol_names=np.asarray(PARAMETER_NAMES),
        sobol_S1=np.asarray(sobol_result["S1"]),
        sobol_S1_conf=np.asarray(sobol_result["S1_conf"]),
        sobol_ST=np.asarray(sobol_result["ST"]),
        sobol_ST_conf=np.asarray(sobol_result["ST_conf"]),
        sobol_convergence_N=convergence["N"],
        sobol_convergence_S1=convergence["S1"],
        sobol_convergence_S1_conf=convergence["S1_conf"],
    )

    print("\n" + "=" * 72)
    print("FINAL HELD-OUT TEST RESULTS")
    print("=" * 72)
    for row in final_test_rows:
        print(
            f"{row['model']:10s} | strategy={row['strategy_selected_on_validation']:8s} "
            f"| N={row['n_train_selected_on_validation']:3d} "
            f"| rel. L2={row['relative_l2_global']:.6e} "
            f"| mean case={row['relative_l2_case_mean']:.6e}"
        )

    print("\nSobol indices (domain-averaged pressure):")
    for i, name in enumerate(PARAMETER_NAMES):
        print(
            f"{name:12s} S1={sobol_result['S1'][i]:+.5f} "
            f"± {sobol_result['S1_conf'][i]:.5f}; "
            f"ST={sobol_result['ST'][i]:+.5f} ± {sobol_result['ST_conf'][i]:.5f}"
        )

    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
    print("Review summary.txt and final_heldout_test_results.csv before reporting results.")


if __name__ == "__main__":
    main()
