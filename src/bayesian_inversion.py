#!/usr/bin/env python
"""
Bayesian parameter inference with the POD-GPR Darcy surrogate
==============================================================

Question
--------
Given noisy pressure readings at a few sensors, which permeability-pattern
parameters (amplitude, angle) produced them -- and does using the surrogate
instead of the full solver distort the answer?

Why this fits the rest of the study
-----------------------------------
Earlier experiments measured surrogate accuracy as a *field error*.  Here the
same surrogate is judged by a downstream task: does its error bias or
over-sharpen a Bayesian posterior?  The POD ranks 8 / 32 / 64 are compared, so
the rank findings are tested in an inference setting too.

What is computed
----------------
* Unknowns: u = (logk_amp, pattern_angle) with a uniform prior on the study box.
  logk_mean is not inferred: pressure does not depend on it (see README).
* Data:  y = p_full(sensors; u_true) + N(0, sigma^2 I)   (generated with the FULL solver)
* Posteriors (all by MCMC, uniform prior):
    - "full solver"          reference (exact model, no surrogate)
    - "surrogate, naive"     likelihood ignores surrogate error
    - "surrogate, discrepancy"  Kennedy-O'Hagan style: surrogate error is modelled
                             as Gaussian with mean/covariance estimated from the
                             validation set at the sensors
* Checks:
    - case studies (3 truths x 3 noise levels): surrogate vs full-solver posterior
    - calibration over many truths drawn from the prior (simulation-based
      calibration style): coverage of credible intervals and PIT histograms
    - MCMC diagnostics (split R-hat, ESS, acceptance) and a multimodality flag

Outputs (in --out, default bayesian_inversion_results/)
-------------------------------------------------------
    figures/*.png, bayesian_inversion_results.xlsx, tables/*.csv, summary.md

Usage (from the folder that contains darcy.py and experiment_a.py)
------------------------------------------------------------------
    PYTHONPATH=src python src/bayesian_inversion.py --quick     # ~2-4 min check
    PYTHONPATH=src python src/bayesian_inversion.py             # full run

Honest limits (also written to summary.md)
------------------------------------------
* Synthetic data from the same solver family: this tests the surrogate, not a
  real-world measurement model.  Noise is i.i.d. Gaussian and known.
* The discrepancy model is stationary (one covariance for the whole parameter
  box) although surrogate error varies in space.
* Reference chains are short; check R-hat/ESS in the Diagnostics sheet.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.interpolate import RectBivariateSpline  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.stats import gaussian_kde, wasserstein_distance  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

import experiment_a as ea  # noqa: E402
from darcy import solve_darcy  # noqa: E402
from pod_rank_sensitivity import fit_rank_limited_pod  # noqa: E402

PHYS_LOW = ea.LOW[1:].copy()  # amp, angle
PHYS_SPAN = (ea.HIGH - ea.LOW)[1:].copy()
PARAM_LABELS = ["logk_amp", "pattern_angle (deg)"]

# validated categorical slots (dataviz reference palette) + ink
COL = {"ref": "#0b0b0b", "naive": "#eb6834", "disc": "#2a78d6", "grid": "#e4e3df", "muted": "#52514e"}
RANK_LS = ["-", "--", ":"]
KIND_LABEL = {"naive": "surrogate, naive", "disc": "surrogate, with discrepancy", "ref": "full solver (reference)"}

CASES = {
    "A_well_identified": (1.00, 60.0),
    "B_low_amplitude": (0.35, 120.0),
    "C_near_boundary": (1.30, 170.0),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_phys(u: np.ndarray) -> np.ndarray:
    return PHYS_LOW + np.asarray(u) * PHYS_SPAN


def to_unit(p: np.ndarray) -> np.ndarray:
    return (np.asarray(p) - PHYS_LOW) / PHYS_SPAN


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def make_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.quick:
        cfg = dict(
            nx=24, n_train=64, n_val=48, ranks=(4, 8, 12), grid=(41, 61), n_side=4,
            sigmas_case=(1e-2, 1e-3, 1e-4), sigmas_cal=(1e-2, 1e-3, 1e-4), n_cal=16,
            chains=4, warm1=150, warm2=150, n_sample=600,
            ref_chains=4, ref_warm=120, ref_sample=300, ref_sigmas=(1e-2, 1e-3), ref_cases=("A_well_identified",),
        )
    else:
        cfg = dict(
            nx=ea.NX, n_train=ea.FINAL_N, n_val=ea.N_VALIDATION, ranks=(8, 32, 64), grid=(131, 181), n_side=6,
            sigmas_case=(1e-2, 1e-3, 1e-4), sigmas_cal=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4), n_cal=100,
            chains=4, warm1=400, warm2=400, n_sample=2000,
            ref_chains=4, ref_warm=250, ref_sample=800, ref_sigmas=(1e-2, 1e-3, 1e-4), ref_cases=tuple(CASES),
        )
    if args.ranks:
        cfg["ranks"] = tuple(int(r) for r in args.ranks.split(","))
    if args.cal_cases:
        cfg["n_cal"] = int(args.cal_cases)
    cfg["quick"] = bool(args.quick)
    cfg["seed"] = int(args.seed)
    cfg["workers"] = int(args.workers)
    cfg["skip_reference"] = bool(args.skip_reference)
    return cfg


# --------------------------------------------------------------------------- #
# Full solver at sensors (also used inside worker processes)
# --------------------------------------------------------------------------- #
def sensor_indices(nx: int, n_side: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.round(np.linspace(0.12, 0.88, n_side) * (nx - 1)).astype(int)
    jj, ii = np.meshgrid(pos, pos, indexing="ij")
    idx = (jj * nx + ii).ravel()
    xy = np.column_stack([((ii.ravel() + 0.5) / nx), ((jj.ravel() + 0.5) / nx)])
    return idx, xy


def full_sensor_values(u2: np.ndarray, nx: int, idx: np.ndarray) -> np.ndarray:
    p_amp, p_ang = to_phys(u2)
    p = solve_darcy(nx=nx, ny=nx, params=(0.0, float(p_amp), float(p_ang)),
                    p_left=ea.P_LEFT, p_right=ea.P_RIGHT)[0]
    return np.asarray(p).ravel()[idx]


# --------------------------------------------------------------------------- #
# MCMC (vectorised over cases x chains) and diagnostics
# --------------------------------------------------------------------------- #
def run_mcmc(loglike, x0: np.ndarray, cov0: np.ndarray, warm1: int, warm2: int, n_sample: int,
             rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Random-walk Metropolis, uniform prior on the unit square.

    x0: (n_cases, chains, 2); cov0: (n_cases, 2, 2) starting proposal covariance.
    loglike(X) -> (n_cases, chains).  Warm-up adapts the proposal scale (target
    acceptance 0.3) and re-estimates the proposal covariance from the previous
    block; the sampling block uses a frozen proposal, so the retained draws are a
    valid Metropolis chain.
    """
    nc, C, d = x0.shape
    x = np.clip(x0, 1e-6, 1 - 1e-6).copy()

    def lp(X):
        inside = np.all((X >= 0.0) & (X <= 1.0), axis=-1)
        ll = loglike(np.clip(X, 0.0, 1.0))
        return np.where(inside, ll, -np.inf)

    ll = lp(x)
    scale = np.full(nc, 2.38 / np.sqrt(d))
    cov = cov0.copy()

    def block(n: int, adapt: bool, Lc: np.ndarray):
        nonlocal x, ll, scale
        out = np.empty((n, nc, C, d))
        acc_tot = np.zeros(nc)
        acc_win = np.zeros(nc)
        for t in range(n):
            z = rng.standard_normal((nc, C, d))
            prop = x + scale[:, None, None] * np.einsum("nij,ncj->nci", Lc, z)
            llp = lp(prop)
            with np.errstate(invalid="ignore"):
                acc = np.log(rng.random((nc, C))) < (llp - ll)
            x = np.where(acc[..., None], prop, x)
            ll = np.where(acc, llp, ll)
            a = acc.mean(axis=1)
            acc_tot += a
            acc_win += a
            if adapt and (t + 1) % 50 == 0:
                scale = scale * np.exp(acc_win / 50.0 - 0.3)
                acc_win[:] = 0
            out[t] = x
        return out, acc_tot / n

    def chol(c):
        return np.linalg.cholesky(c + 1e-16 * np.eye(d)[None])

    for nb in (warm1, warm2, warm2):
        samp, _ = block(nb, True, chol(cov))
        tail = samp[nb // 2:].transpose(1, 0, 2, 3).reshape(nc, -1, d)
        for i in range(nc):
            c = np.cov(tail[i].T)
            if np.all(np.isfinite(c)) and np.linalg.eigvalsh(c).min() > 1e-14:
                cov[i] = c
        scale = np.full(nc, 2.38 / np.sqrt(d)) * np.clip(scale / (2.38 / np.sqrt(d)), 0.3, 3.0)
    samples, acc = block(n_sample, False, chol(cov))
    return {"samples": samples.transpose(1, 2, 0, 3), "acc": acc}  # (nc, C, n, d)


def rhat_ess(chains: np.ndarray) -> tuple[float, float]:
    """Split R-hat and ESS (Geyer initial positive sequence) for chains of shape (C, n)."""
    C, n = chains.shape
    half = n // 2
    sp = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    m, nn = sp.shape
    W = sp.var(axis=1, ddof=1).mean()
    if not np.isfinite(W) or W <= 0:
        return float("nan"), float("nan")
    B = nn * sp.mean(axis=1).var(ddof=1)
    var_hat = (nn - 1) / nn * W + B / nn
    rhat = float(np.sqrt(var_hat / W))
    x = sp - sp.mean(axis=1, keepdims=True)
    f = np.fft.rfft(x, n=2 * nn, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :nn] / nn
    rho = 1.0 - (W - acov.mean(axis=0)) / var_hat
    rho[0] = 1.0
    s, t = 0.0, 1
    while t + 1 < nn:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        s += pair
        t += 2
    tau = 1.0 + 2.0 * (s + 0.0) - 0.0
    tau = max(tau, 1.0 / np.log10(m * nn)) if np.isfinite(tau) else 1.0
    return rhat, float(m * nn / tau)


# --------------------------------------------------------------------------- #
# Surrogate on sensors, tabulated + cubic spline
# --------------------------------------------------------------------------- #
class SensorSurrogate:
    def __init__(self, u1: np.ndarray, u2: np.ndarray, T: np.ndarray):
        self.u1, self.u2, self.T = u1, u2, T
        self.g1, self.g2, self.ns = len(u1), len(u2), T.shape[1]
        self.splines = [RectBivariateSpline(u1, u2, T[:, s].reshape(self.g1, self.g2), kx=3, ky=3, s=0)
                        for s in range(self.ns)]

    def eval(self, U: np.ndarray) -> np.ndarray:
        out = np.empty((len(U), self.ns))
        for s, spl in enumerate(self.splines):
            out[:, s] = spl.ev(U[:, 0], U[:, 1])
        return out

    def jac(self, u: np.ndarray) -> np.ndarray:
        J = np.empty((self.ns, 2))
        for s, spl in enumerate(self.splines):
            J[s, 0] = spl.ev(u[0], u[1], dx=1)
            J[s, 1] = spl.ev(u[0], u[1], dy=1)
        return J

    def grid_loglik(self, y: np.ndarray, lik: "Likelihood") -> np.ndarray:
        R = (y - lik.m - self.T) @ lik.W.T
        return (-0.5 * np.sum(R * R, axis=1)).reshape(self.g1, self.g2)


class Likelihood:
    """Gaussian likelihood y ~ N(s(u) + m, sigma^2 I + Sigma_extra)."""

    def __init__(self, sigma: float, ns: int, m: np.ndarray | None = None, Sigma_extra: np.ndarray | None = None):
        Sigma = sigma**2 * np.eye(ns)
        if Sigma_extra is not None:
            Sigma = Sigma + Sigma_extra
        L = np.linalg.cholesky(Sigma)
        self.W = solve_triangular(L, np.eye(ns), lower=True)
        self.m = np.zeros(ns) if m is None else m
        self.sigma = sigma


def find_modes(sur: SensorSurrogate, lik: Likelihood, y: np.ndarray, max_starts: int = 4) -> dict[str, Any]:
    ll = sur.grid_loglik(y, lik)
    mf = maximum_filter(ll, size=3, mode="nearest")
    peaks = np.argwhere(ll == mf)
    order = np.argsort(-ll[tuple(peaks.T)])[:max_starts]

    def fun(u):
        return lik.W @ (y - lik.m - sur.eval(u[None])[0])

    def jac(u):
        return -lik.W @ sur.jac(u)

    sols = []
    for i, j in peaks[order]:
        u0 = np.array([sur.u1[i], sur.u2[j]])
        u0 = np.clip(u0, 1e-6, 1 - 1e-6)
        try:
            r = least_squares(fun, u0, jac=jac, bounds=(0.0, 1.0), x_scale="jac", method="trf",
                              xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=80)
            sols.append((r.x, -r.cost, r.jac))
        except Exception:  # pragma: no cover - defensive
            continue
    if not sols:
        i, j = np.unravel_index(np.argmax(ll), ll.shape)
        u = np.array([sur.u1[i], sur.u2[j]])
        sols = [(u, ll[i, j], jac(u))]
    sols.sort(key=lambda s: -s[1])
    distinct = []
    for u, lp_, J in sols:
        if all(np.linalg.norm(u - v[0]) > 1e-3 for v in distinct):
            distinct.append((u, lp_, J))
    best = distinct[0]
    significant = [s for s in distinct if s[1] >= best[1] - 4.6]
    prec = best[2].T @ best[2] + 4.0 * np.eye(2)  # weak regularisation (proposal only)
    cov = np.linalg.inv(prec)
    return {"u_map": best[0], "cov": cov, "modes": [s[0] for s in significant], "n_modes": len(significant)}


def make_loglike(sur: SensorSurrogate, lik: Likelihood, Y: np.ndarray):
    nc, ns = Y.shape

    def f(X):
        C = X.shape[1]
        S = sur.eval(X.reshape(-1, 2)).reshape(nc, C, ns)
        R = (Y[:, None, :] - lik.m - S) @ lik.W.T
        return -0.5 * np.sum(R * R, axis=-1)

    return f


def run_arm(sur: SensorSurrogate, lik: Likelihood, Y: np.ndarray, cfg: dict[str, Any],
            rng: np.random.Generator) -> dict[str, Any]:
    """Posterior sampling for every data vector in Y with one surrogate/likelihood."""
    nc = Y.shape[0]
    C = cfg["chains"]
    x0 = np.empty((nc, C, 2))
    cov0 = np.empty((nc, 2, 2))
    n_modes = np.empty(nc, dtype=int)
    umap = np.empty((nc, 2))
    for i in range(nc):
        md = find_modes(sur, lik, Y[i])
        Lc = np.linalg.cholesky(md["cov"])
        for c in range(C):
            base = md["modes"][c % len(md["modes"])]
            x0[i, c] = base + Lc @ rng.standard_normal(2)
        cov0[i] = md["cov"]
        n_modes[i] = md["n_modes"]
        umap[i] = md["u_map"]
    res = run_mcmc(make_loglike(sur, lik, Y), x0, cov0, cfg["warm1"], cfg["warm2"], cfg["n_sample"], rng)
    S = res["samples"]
    rh = np.full((nc, 2), np.nan)
    es = np.full((nc, 2), np.nan)
    for i in range(nc):
        for j in range(2):
            rh[i, j], es[i, j] = rhat_ess(S[i, :, :, j])
    return {"samples": S, "acc": res["acc"], "rhat": rh, "ess": es, "n_modes": n_modes, "u_map": umap}


# --------------------------------------------------------------------------- #
# Full-solver reference posterior (worker; runs in a separate process)
# --------------------------------------------------------------------------- #
def ref_worker(job: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    nx, idx, y, sigma = job["nx"], job["idx"], job["y"], job["sigma"]
    nsolve = [0]

    def s_full(u):
        nsolve[0] += 1
        return full_sensor_values(np.clip(u, 0.0, 1.0), nx, idx)

    def fun(u):
        return (y - s_full(u)) / sigma

    def jac(u):
        h = 2e-5
        J = np.empty((len(y), 2))
        for k in range(2):
            e = np.zeros(2)
            e[k] = h
            up = np.clip(u + e, 0, 1)
            um = np.clip(u - e, 0, 1)
            J[:, k] = -(s_full(up) - s_full(um)) / (up[k] - um[k]) / sigma
        return J

    sols = []
    for u0 in job["inits"]:
        u0 = np.clip(np.asarray(u0, dtype=float), 1e-6, 1 - 1e-6)
        r = least_squares(fun, u0, jac=jac, bounds=(0.0, 1.0), x_scale="jac", method="trf",
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=40)
        sols.append((r.x, -r.cost, r.jac))
    sols.sort(key=lambda s: -s[1])
    distinct = []
    for u, lp_, J in sols:
        if all(np.linalg.norm(u - v[0]) > 1e-3 for v in distinct):
            distinct.append((u, lp_, J))
    best = distinct[0]
    significant = [s for s in distinct if s[1] >= best[1] - 4.6]
    cov = np.linalg.inv(best[2].T @ best[2] + 4.0 * np.eye(2))
    Lc = np.linalg.cholesky(cov)
    rng = np.random.default_rng(job["seed"])
    C = job["chains"]
    x0 = np.empty((1, C, 2))
    for c in range(C):
        x0[0, c] = significant[c % len(significant)][0] + Lc @ rng.standard_normal(2)

    def loglike(X):
        out = np.empty(X.shape[:2])
        for c in range(X.shape[1]):
            r = (y - s_full(X[0, c])) / sigma
            out[0, c] = -0.5 * float(r @ r)
        return out

    res = run_mcmc(loglike, x0, cov[None], job["warm"], job["warm"], job["n_sample"], rng)
    S = res["samples"]
    rh = [rhat_ess(S[0, :, :, j])[0] for j in range(2)]
    es = [rhat_ess(S[0, :, :, j])[1] for j in range(2)]
    return {"key": job["key"], "samples": S[0], "acc": float(res["acc"][0]), "rhat": rh, "ess": es,
            "n_modes": len(significant), "u_map": best[0], "n_solves": nsolve[0],
            "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# Posterior summaries and comparisons
# --------------------------------------------------------------------------- #
def gauss_kl(m0, S0, m1, S1) -> float:
    """KL( N(m0,S0) || N(m1,S1) )."""
    S1i = np.linalg.inv(S1)
    d = len(m0)
    dm = m1 - m0
    return float(0.5 * (np.trace(S1i @ S0) + dm @ S1i @ dm - d + np.log(np.linalg.det(S1) / np.linalg.det(S0))))


def joint_p(S: np.ndarray, truth_u: np.ndarray) -> float:
    mu = S.mean(0)
    Ci = np.linalg.inv(np.cov(S.T) + 1e-30 * np.eye(2))
    d2s = np.einsum("ni,ij,nj->n", S - mu, Ci, S - mu)
    dt = (truth_u - mu) @ Ci @ (truth_u - mu)
    return float(np.mean(d2s < dt))


def post_summary(S_chains: np.ndarray, truth_u: np.ndarray) -> dict[str, float]:
    """S_chains: (C, n, 2) in unit space.  Returns physical-unit summary."""
    S = S_chains.reshape(-1, 2)
    P = to_phys(S)
    tp = to_phys(truth_u)
    out: dict[str, float] = {}
    for j, nm in enumerate(["amp", "angle"]):
        q = np.quantile(P[:, j], [0.05, 0.25, 0.5, 0.75, 0.95])
        out[f"{nm}_truth"] = float(tp[j])
        out[f"{nm}_mean"] = float(P[:, j].mean())
        out[f"{nm}_sd"] = float(P[:, j].std(ddof=1))
        out[f"{nm}_q05"], out[f"{nm}_q25"], out[f"{nm}_q50"], out[f"{nm}_q75"], out[f"{nm}_q95"] = map(float, q)
        out[f"{nm}_pit"] = float(np.mean(P[:, j] < tp[j]))
        out[f"{nm}_in50"] = float(out[f"{nm}_q25"] <= tp[j] <= out[f"{nm}_q75"])
        out[f"{nm}_in90"] = float(out[f"{nm}_q05"] <= tp[j] <= out[f"{nm}_q95"])
        out[f"{nm}_z"] = float((out[f"{nm}_mean"] - tp[j]) / max(out[f"{nm}_sd"], 1e-300))
    pj = joint_p(S, truth_u)
    out["joint_p"] = pj
    out["joint_in90"] = float(pj <= 0.9)
    out["joint_in50"] = float(pj <= 0.5)
    return out


def compare_to_ref(S_arm: np.ndarray, S_ref: np.ndarray) -> dict[str, float]:
    A, R = S_arm.reshape(-1, 2), S_ref.reshape(-1, 2)
    mA, mR = A.mean(0), R.mean(0)
    cA, cR = np.cov(A.T), np.cov(R.T)
    sdA, sdR = A.std(0, ddof=1), R.std(0, ddof=1)
    cRi = np.linalg.inv(cR + 1e-30 * np.eye(2))
    out = {
        "shift_mahal_ref": float(np.sqrt((mA - mR) @ cRi @ (mA - mR))),
        "shift_amp_in_ref_sd": float((mA[0] - mR[0]) / sdR[0]),
        "shift_angle_in_ref_sd": float((mA[1] - mR[1]) / sdR[1]),
        "sd_ratio_amp": float(sdA[0] / sdR[0]),
        "sd_ratio_angle": float(sdA[1] / sdR[1]),
        "kl_ref_to_arm": gauss_kl(mR, cR, mA, cA),
        "kl_arm_to_ref": gauss_kl(mA, cA, mR, cR),
    }
    for j, nm in enumerate(["amp", "angle"]):
        out[f"w1_{nm}_in_ref_sd"] = float(wasserstein_distance(A[::4, j], R[::4, j]) / sdR[j])
    return out


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def prepare_data(cfg: dict[str, Any], cache: Path) -> dict[str, Any]:
    nx = cfg["nx"]
    ea.NX = nx
    ea.NY = nx
    key = f"nx{nx}_N{cfg['n_train']}_V{cfg['n_val']}"
    f = cache / f"data_{key}.npz"
    if f.exists():
        z = np.load(f)
        log(f"Loaded cached solver data ({key}).")
        return {k: z[k] for k in z.files}
    log(f"Running {cfg['n_train'] + cfg['n_val']} full solves for training/validation ({nx}x{nx})...")
    U_tr = ea.make_lhs(cfg["n_train"], ea.FINAL_TRAIN_SEED)
    U_va = ea.make_lhs(cfg["n_val"], ea.VALIDATION_SEED)
    S_tr, _, _ = ea.solve_many(U_tr)
    S_va, _, _ = ea.solve_many(U_va)
    np.savez(f, U_tr=U_tr, U_va=U_va, S_tr=S_tr, S_va=S_va)
    return {"U_tr": U_tr, "U_va": U_va, "S_tr": S_tr, "S_va": S_va}


def build_surrogate(rank: int, data: dict[str, Any], idx: np.ndarray, cfg: dict[str, Any],
                    cache: Path) -> dict[str, Any]:
    g1, g2 = cfg["grid"]
    f = cache / f"table_nx{cfg['nx']}_N{cfg['n_train']}_V{cfg['n_val']}_r{rank}_g{g1}x{g2}_s{len(idx)}.npz"
    if f.exists():
        z = np.load(f)
        log(f"Loaded cached surrogate table (rank {rank}).")
        info = {k: z[k] for k in z.files}
    else:
        t0 = time.perf_counter()
        cols = list(ea.PRESSURE_INPUT_COLUMNS)
        X_tr = data["U_tr"][:, cols]
        X_va = data["U_va"][:, cols]
        pod = fit_rank_limited_pod(data["S_tr"], rank)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predict, _, _ = ea.fit_candidate("GPR", {"kernel_name": "RBF"}, X_tr, pod["coefficients"], ea.MASTER_SEED)
        fit_s = time.perf_counter() - t0
        u1, u2 = np.linspace(0, 1, g1), np.linspace(0, 1, g2)
        G = np.column_stack([a.ravel() for a in np.meshgrid(u1, u2, indexing="ij")])
        coef = predict(G)
        T = pod["mean_field"][idx] + coef @ pod["phi"][idx].T
        coef_v = predict(X_va)
        P_va = pod["mean_field"] + coef_v @ pod["phi"].T
        val_sens = P_va[:, idx]
        rel = float(np.linalg.norm(P_va - data["S_va"]) / np.linalg.norm(data["S_va"]))
        info = {"T": T, "val_sens_pred": val_sens, "val_rel_l2": np.array(rel),
                "fit_seconds": np.array(fit_s), "captured_energy": np.array(pod["captured_energy"])}
        np.savez(f, **info)
        log(f"Rank {rank}: GP bank fitted in {fit_s:.0f}s, validation field rel. L2 = {rel:.2e}")
    u1, u2 = np.linspace(0, 1, g1), np.linspace(0, 1, g2)
    sur = SensorSurrogate(u1, u2, info["T"])
    resid = data["S_va"][:, idx] - info["val_sens_pred"]
    lw = LedoitWolf().fit(resid)
    return {
        "rank": rank, "sur": sur, "resid": resid, "m_delta": resid.mean(axis=0), "Sigma_delta": lw.covariance_,
        "val_rel_l2": float(info["val_rel_l2"]), "fit_seconds": float(info["fit_seconds"]),
        "captured_energy": float(info["captured_energy"]),
        "rms_sensor_resid": float(np.sqrt(np.mean(resid**2))),
        "rms_sensor_resid_centered": float(np.sqrt(np.mean((resid - resid.mean(0))**2))),
    }


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
def self_tests(cfg: dict[str, Any], surs: list[dict[str, Any]], data: dict[str, Any], idx: np.ndarray) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)

    # 1. sampler on a correlated Gaussian
    m = np.array([0.4, 0.6])
    sd = np.array([0.05, 0.03])
    rho = 0.6
    cov = np.array([[sd[0]**2, rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], sd[1]**2]])
    Ci = np.linalg.inv(cov)

    def ll(X):
        d = X - m
        return -0.5 * np.einsum("nci,ij,ncj->nc", d, Ci, d)

    x0 = m + rng.standard_normal((1, 4, 2)) * sd * 2
    res = run_mcmc(ll, x0, np.array([cov * 4]), 300, 300, 4000, rng)
    S = res["samples"][0].reshape(-1, 2)
    em = np.abs(S.mean(0) - m) / sd
    ec = np.abs(np.cov(S.T) / cov - 1).max()
    rows.append(("MCMC recovers a known 2D Gaussian", f"max mean error {em.max():.3f} sd, max cov error {ec:.1%}",
                 bool(em.max() < 0.15 and ec < 0.2)))

    # 2. spline table vs the full solver at the sensors (interpolation + GP error); uses validation points
    s0 = surs[len(surs) // 2]
    Uv = data["U_va"][:10][:, list(ea.PRESSURE_INPUT_COLUMNS)]
    spline_vals = s0["sur"].eval(Uv)
    gp_err = np.abs(spline_vals - data["S_va"][:10][:, idx]).max()
    rows.append((f"Spline table vs full solver at sensors (rank {s0['rank']}, 10 validation points)",
                 f"max abs error {gp_err:.2e}", bool(gp_err < 0.05)))

    # 3. worker solver path equals batch solver path
    u = Uv[0]
    a = full_sensor_values(u, cfg["nx"], idx)
    b = data["S_va"][0][idx]
    rows.append(("Single-point solver path equals batch solver path", f"max abs diff {np.abs(a - b).max():.1e}",
                 bool(np.abs(a - b).max() < 1e-10)))

    # 4. spline Jacobian vs finite differences
    u = np.array([0.4, 0.55])
    J = s0["sur"].jac(u)
    h = 1e-6
    Jfd = np.column_stack([(s0["sur"].eval((u + e)[None]) - s0["sur"].eval((u - e)[None]))[0] / (2 * h)
                           for e in (np.array([h, 0]), np.array([0, h]))])
    err = np.abs(J - Jfd).max() / max(np.abs(Jfd).max(), 1e-12)
    rows.append(("Spline Jacobian equals finite differences", f"relative error {err:.1e}", bool(err < 1e-4)))

    df = pd.DataFrame(rows, columns=["test", "result", "passed"])
    for _, r in df.iterrows():
        log(f"  self-test [{'ok' if r['passed'] else 'FAIL'}] {r['test']}: {r['result']}")
    if not df["passed"].all():
        raise SystemExit("A self-test failed; stopping before any results are produced.")
    return df


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": COL["muted"], "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
        "xtick.color": COL["muted"], "ytick.color": COL["muted"], "axes.grid": True, "grid.color": COL["grid"],
        "grid.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    })


def hpd_contours(ax, S_u: np.ndarray, xlim, ylim, color, ls="-") -> None:
    P = to_phys(S_u.reshape(-1, 2))
    P = P[:: max(1, len(P) // 2500)]
    try:
        kde = gaussian_kde(P.T)
        gx, gy = np.linspace(*xlim, 130), np.linspace(*ylim, 130)
        GX, GY = np.meshgrid(gx, gy)
        Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        zs = np.sort(Z.ravel())[::-1]
        cs = np.cumsum(zs) / zs.sum()
        lv = sorted({float(zs[min(np.searchsorted(cs, m), len(zs) - 1)]) for m in (0.5, 0.9)})
        if len(lv) == 2:
            ax.contour(GX, GY, Z, levels=lv, colors=[color], linestyles=ls, linewidths=[1.0, 1.8])
    except Exception:
        ax.plot(P[:, 0], P[:, 1], ".", ms=1.5, color=color, alpha=0.4)


def panel_limits(sample_sets, truth_p):
    allp = np.vstack([to_phys(s.reshape(-1, 2)) for s in sample_sets])
    lo = np.quantile(allp, 0.003, axis=0)
    hi = np.quantile(allp, 0.997, axis=0)
    lo = np.minimum(lo, truth_p)
    hi = np.maximum(hi, truth_p)
    pad = 0.25 * np.maximum(hi - lo, 1e-9)
    lo = np.maximum(lo - pad, PHYS_LOW)
    hi = np.minimum(hi + pad, PHYS_LOW + PHYS_SPAN)
    return (lo[0], hi[0]), (lo[1], hi[1])


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small, fast configuration to test the workflow")
    ap.add_argument("--out", default="bayesian_inversion_results")
    ap.add_argument("--ranks", default="", help="comma list of POD ranks, e.g. 8,32,64")
    ap.add_argument("--cal-cases", default="", help="number of prior-drawn truths for calibration")
    ap.add_argument("--workers", type=int, default=min(2, max(1, (mp.cpu_count() or 1))),
                    help="processes for the full-solver reference chains")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--skip-reference", action="store_true", help="skip the (slow) full-solver reference posteriors")
    ap.add_argument("--fresh", action="store_true", help="ignore cached solver data / tables / reference chains")
    args = ap.parse_args()
    cfg = make_config(args)

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    if args.fresh:
        for f in cache.glob("*.npz"):
            f.unlink()
    style()
    T_start = time.perf_counter()
    rng = np.random.default_rng(cfg["seed"])
    log(f"Configuration: {cfg}")

    # ---- data, sensors, surrogates -------------------------------------------------
    data = prepare_data(cfg, cache)
    idx, sensor_xy = sensor_indices(cfg["nx"], cfg["n_side"])
    ns = len(idx)
    surs = [build_surrogate(r, data, idx, cfg, cache) for r in cfg["ranks"]]
    mid = len(surs) // 2
    log("Running self-tests...")
    st = self_tests(cfg, surs, data, idx)

    surr_rows = [{
        "pod_rank": s["rank"], "captured_energy": s["captured_energy"], "gp_fit_seconds": s["fit_seconds"],
        "validation_field_rel_L2": s["val_rel_l2"], "sensor_residual_RMS": s["rms_sensor_resid"],
        "sensor_residual_RMS_bias_removed": s["rms_sensor_resid_centered"],
        "sensor_bias_RMS": float(np.sqrt(np.mean(s["m_delta"]**2))),
    } for s in surs]
    df_surr = pd.DataFrame(surr_rows)
    log("Surrogate quality at the sensors:\n" + df_surr.to_string(index=False))

    arms = []  # (rank index, kind)
    for si, s in enumerate(surs):
        arms.append((si, "naive"))
        arms.append((si, "disc"))

    def lik_for(si: int, kind: str, sigma: float) -> Likelihood:
        s = surs[si]
        if kind == "naive":
            return Likelihood(sigma, ns)
        return Likelihood(sigma, ns, m=s["m_delta"], Sigma_extra=s["Sigma_delta"])

    # ---- Experiment 1: case studies ----------------------------------------------
    log("Experiment 1: case studies (3 truths x noise levels)")
    case_names = list(CASES)
    U_case = np.array([to_unit(np.array(CASES[c])) for c in case_names])
    Yfull_case = np.array([full_sensor_values(u, cfg["nx"], idx) for u in U_case])
    eps_case = rng.standard_normal(Yfull_case.shape)
    case_samples: dict[tuple, np.ndarray] = {}
    case_rows, diag_rows = [], []
    case_modes: dict[tuple, list] = {}
    for sigma in cfg["sigmas_case"]:
        Y = Yfull_case + sigma * eps_case
        for si, kind in arms:
            t0 = time.perf_counter()
            res = run_arm(surs[si]["sur"], lik_for(si, kind, sigma), Y, cfg, rng)
            for ci, cn in enumerate(case_names):
                case_samples[(cn, sigma, si, kind)] = res["samples"][ci]
                row = post_summary(res["samples"][ci], U_case[ci])
                row.update({"case": cn, "sigma": sigma, "pod_rank": surs[si]["rank"], "likelihood": kind,
                            "n_modes": int(res["n_modes"][ci]), "rhat_max": float(np.nanmax(res["rhat"][ci])),
                            "ess_min": float(np.nanmin(res["ess"][ci])), "acceptance": float(res["acc"][ci]),
                            "u_map_amp": float(to_phys(res["u_map"][ci])[0]),
                            "u_map_angle": float(to_phys(res["u_map"][ci])[1])})
                case_rows.append(row)
                if si == mid and kind == "naive":
                    case_modes[(cn, sigma)] = [res["u_map"][ci]]
            log(f"  sigma={sigma:g} rank={surs[si]['rank']} {kind}: {time.perf_counter() - t0:.1f}s")
    df_case = pd.DataFrame(case_rows)

    # ---- Experiment 1b: full-solver reference -----------------------------------------
    ref: dict[tuple, dict[str, Any]] = {}
    if not cfg["skip_reference"]:
        jobs = []
        for cn in cfg["ref_cases"]:
            ci = case_names.index(cn)
            for sigma in cfg["ref_sigmas"]:
                key = f"{cn}_s{sigma:g}"
                f = cache / f"ref_nx{cfg['nx']}_{key}_{cfg['ref_warm']}_{cfg['ref_sample']}_{cfg['seed']}.npz"
                if f.exists():
                    z = np.load(f, allow_pickle=True)
                    ref[(cn, sigma)] = {k: z[k] for k in z.files}
                    continue
                inits = [case_modes[(cn, sigma)][0]]
                inits += [to_unit(np.array([row["amp_mean"], row["angle_mean"]])) for row in case_rows
                          if row["case"] == cn and row["sigma"] == sigma and row["pod_rank"] == surs[mid]["rank"]]
                jobs.append({"key": key, "cache": str(f), "case": cn, "sigma": sigma, "nx": cfg["nx"], "idx": idx,
                             "y": Yfull_case[ci] + sigma * eps_case[ci], "inits": inits, "chains": cfg["ref_chains"],
                             "warm": cfg["ref_warm"], "n_sample": cfg["ref_sample"], "seed": cfg["seed"] + 17 * (ci + 1)})
        if jobs:
            log(f"Running {len(jobs)} full-solver reference posteriors with {cfg['workers']} worker(s); "
                f"each uses ~{(3 * cfg['ref_warm'] + cfg['ref_sample']) * cfg['ref_chains']} solves (about 0.04 s each)...")
            t0 = time.perf_counter()
            if cfg["workers"] > 1 and len(jobs) > 1:
                with mp.get_context("spawn").Pool(cfg["workers"]) as pool:
                    results = []
                    for r in pool.imap_unordered(ref_worker, jobs):
                        results.append(r)
                        log(f"  reference done: {r['key']} ({r['seconds']:.0f}s, acc {r['acc']:.2f}, "
                            f"R-hat {max(r['rhat']):.3f}, ESS {min(r['ess']):.0f})")
            else:
                results = []
                for j in jobs:
                    r = ref_worker(j)
                    results.append(r)
                    log(f"  reference done: {r['key']} ({r['seconds']:.0f}s, acc {r['acc']:.2f}, "
                        f"R-hat {max(r['rhat']):.3f}, ESS {min(r['ess']):.0f})")
            by_key = {j["key"]: j for j in jobs}
            for r in results:
                j = by_key[r["key"]]
                np.savez(j["cache"], **{k: np.array(v) for k, v in r.items() if k != "key"})
                ref[(j["case"], j["sigma"])] = {k: np.array(v) for k, v in r.items() if k != "key"}
            log(f"Reference chains finished in {time.perf_counter() - t0:.0f}s")

    cmp_rows = []
    for (cn, sigma), r in ref.items():
        ci = case_names.index(cn)
        row = post_summary(r["samples"], U_case[ci])
        row.update({"case": cn, "sigma": sigma, "source": "full solver", "acceptance": float(r["acc"]),
                    "rhat_max": float(np.max(r["rhat"])), "ess_min": float(np.min(r["ess"])),
                    "n_modes": int(r["n_modes"]), "n_solves": int(r["n_solves"]), "seconds": float(r["seconds"])})
        diag_rows.append(row)
        for si, kind in arms:
            cmp = compare_to_ref(case_samples[(cn, sigma, si, kind)], r["samples"])
            cmp.update({"case": cn, "sigma": sigma, "pod_rank": surs[si]["rank"], "likelihood": kind})
            cmp_rows.append(cmp)
    df_cmp = pd.DataFrame(cmp_rows)
    df_ref = pd.DataFrame(diag_rows)

    # ---- Experiment 2: calibration over prior truths ------------------------------
    n_cal = cfg["n_cal"]
    log(f"Experiment 2: calibration with {n_cal} truths drawn from the prior")
    U_cal = rng.random((n_cal, 2))
    U3 = np.column_stack([np.full(n_cal, 0.5), U_cal])
    S_cal, _, _ = ea.solve_many(U3)
    Y_cal_clean = S_cal[:, idx]
    eps_cal = rng.standard_normal(Y_cal_clean.shape)
    cal_rows = []
    for sigma in cfg["sigmas_cal"]:
        Y = Y_cal_clean + sigma * eps_cal
        for si, kind in arms:
            t0 = time.perf_counter()
            res = run_arm(surs[si]["sur"], lik_for(si, kind, sigma), Y, cfg, rng)
            for i in range(n_cal):
                row = post_summary(res["samples"][i], U_cal[i])
                row.update({"case_id": i, "sigma": sigma, "pod_rank": surs[si]["rank"], "likelihood": kind,
                            "n_modes": int(res["n_modes"][i]), "rhat_max": float(np.nanmax(res["rhat"][i])),
                            "ess_min": float(np.nanmin(res["ess"][i])), "acceptance": float(res["acc"][i])})
                cal_rows.append(row)
            log(f"  sigma={sigma:g} rank={surs[si]['rank']} {kind}: {time.perf_counter() - t0:.1f}s")
    df_cal = pd.DataFrame(cal_rows)

    # ---- aggregate tables ----------------------------------------------------------
    grp = ["sigma", "pod_rank", "likelihood"]
    agg = df_cal.groupby(grp).agg(
        n=("case_id", "count"),
        joint_cov90=("joint_in90", "mean"), joint_cov50=("joint_in50", "mean"),
        amp_cov90=("amp_in90", "mean"), angle_cov90=("angle_in90", "mean"),
        amp_cov50=("amp_in50", "mean"), angle_cov50=("angle_in50", "mean"),
        amp_sd_mean=("amp_sd", "mean"), angle_sd_mean=("angle_sd", "mean"),
        amp_abs_err_median=("amp_z", lambda z: np.nan), angle_abs_err_median=("angle_z", lambda z: np.nan),
        rhat_gt_1p05=("rhat_max", lambda r: float(np.mean(r > 1.05))),
        ess_lt_200=("ess_min", lambda e: float(np.mean(e < 200))),
        multimodal_share=("n_modes", lambda m: float(np.mean(m > 1))),
        acceptance_mean=("acceptance", "mean"),
    ).reset_index()
    for nm in ("amp", "angle"):
        err = (df_cal[f"{nm}_mean"] - df_cal[f"{nm}_truth"]).abs()
        e = err.groupby([df_cal[c] for c in grp]).median().reset_index(name=f"{nm}_abs_err_median")
        agg = agg.drop(columns=[f"{nm}_abs_err_median"]).merge(e, on=grp)
    z_sd = df_cal.groupby(grp).agg(amp_z_sd=("amp_z", "std"), angle_z_sd=("angle_z", "std")).reset_index()
    agg = agg.merge(z_sd, on=grp)

    # ---- figures -------------------------------------------------------------------
    log("Writing figures...")
    fig_dir = out / "figures"

    # F1: set-up
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    u_demo = U_case[0]
    p_full = solve_darcy(nx=cfg["nx"], ny=cfg["nx"], params=(0.0, *to_phys(u_demo)), p_left=1.0, p_right=0.0)[0]
    im = ax[0].imshow(p_full, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=1)
    ax[0].scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=22, c="#0b0b0b", edgecolors="#fcfcfb", linewidths=0.8)
    ax[0].set_title(f"{ns} pressure sensors on the full-solver field (case A)")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.046, label="pressure")
    ranks_ = [s["rank"] for s in surs]
    ax[1].plot(ranks_, [s["rms_sensor_resid"] for s in surs], "o-", color=COL["disc"], label="surrogate RMS error at sensors")
    for sg in cfg["sigmas_cal"]:
        ax[1].axhline(sg, color=COL["muted"], lw=0.8, ls=":")
        ax[1].text(ranks_[-1], sg, f" noise {sg:g}", va="bottom", ha="right", fontsize=8, color=COL["muted"])
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")
    ax[1].set_xticks(ranks_)
    ax[1].set_xticklabels([str(r) for r in ranks_])
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_xlabel("POD rank")
    ax[1].set_ylabel("pressure units")
    ax[1].set_title("Where surrogate error exceeds measurement noise\nthe naive posterior can no longer be trusted")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_setup_sensors_and_noise.png", dpi=200)
    plt.close(fig)

    # F2: case posteriors (middle rank)
    def case_figure(rank_idx: int, sigmas, fname: str, title: str) -> None:
        nr, nc_ = len(case_names), len(sigmas)
        fig, axes = plt.subplots(nr, nc_, figsize=(3.6 * nc_ + 1, 3.0 * nr), squeeze=False)
        for ri, cn in enumerate(case_names):
            for ci_, sg in enumerate(sigmas):
                ax = axes[ri, ci_]
                sets = [case_samples[(cn, sg, rank_idx, "naive")], case_samples[(cn, sg, rank_idx, "disc")]]
                if (cn, sg) in ref:
                    sets.append(ref[(cn, sg)]["samples"])
                tp = np.array(CASES[cn])
                xl, yl = panel_limits(sets, tp)
                if (cn, sg) in ref:
                    hpd_contours(ax, ref[(cn, sg)]["samples"], xl, yl, COL["ref"], "-")
                hpd_contours(ax, sets[0], xl, yl, COL["naive"], "-")
                hpd_contours(ax, sets[1], xl, yl, COL["disc"], "--")
                ax.plot(*tp, marker="*", ms=12, color="#0b0b0b", mfc="#fcfcfb", mew=1.4, ls="none")
                ax.set_xlim(xl)
                ax.set_ylim(yl)
                if ri == 0:
                    ax.set_title(f"noise {sg:g}")
                if ci_ == 0:
                    ax.set_ylabel(f"{cn}\nangle (deg)")
                if ri == nr - 1:
                    ax.set_xlabel("logk_amp")
        handles = [Line2D([], [], color=COL["ref"], lw=1.8), Line2D([], [], color=COL["naive"], lw=1.8),
                   Line2D([], [], color=COL["disc"], lw=1.8, ls="--"),
                   Line2D([], [], marker="*", color="#0b0b0b", mfc="#fcfcfb", ls="none", ms=10)]
        labels = ["full solver" if ref else "(no reference run)", "surrogate, naive", "surrogate, discrepancy", "true value"]
        fig.legend(handles, labels, loc="lower center", ncol=4)
        fig.suptitle(title, fontweight="bold")
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.savefig(fig_dir / fname, dpi=200)
        plt.close(fig)

    case_figure(mid, cfg["sigmas_case"], "fig2_case_posteriors.png",
                f"Posterior of (amplitude, angle): 50% and 90% regions, POD rank {surs[mid]['rank']}")
    mid_sigma = cfg["sigmas_case"][len(cfg["sigmas_case"]) // 2]
    fig, axes = plt.subplots(len(case_names), len(surs), figsize=(3.6 * len(surs) + 1, 3.0 * len(case_names)), squeeze=False)
    for ri, cn in enumerate(case_names):
        for si, s in enumerate(surs):
            ax = axes[ri, si]
            sets = [case_samples[(cn, mid_sigma, si, "naive")], case_samples[(cn, mid_sigma, si, "disc")]]
            if (cn, mid_sigma) in ref:
                sets.append(ref[(cn, mid_sigma)]["samples"])
            tp = np.array(CASES[cn])
            xl, yl = panel_limits(sets, tp)
            if (cn, mid_sigma) in ref:
                hpd_contours(ax, ref[(cn, mid_sigma)]["samples"], xl, yl, COL["ref"], "-")
            hpd_contours(ax, sets[0], xl, yl, COL["naive"], "-")
            hpd_contours(ax, sets[1], xl, yl, COL["disc"], "--")
            ax.plot(*tp, marker="*", ms=12, color="#0b0b0b", mfc="#fcfcfb", mew=1.4, ls="none")
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            if ri == 0:
                ax.set_title(f"POD rank {s['rank']}")
            if si == 0:
                ax.set_ylabel(f"{cn}\nangle (deg)")
            if ri == len(case_names) - 1:
                ax.set_xlabel("logk_amp")
    fig.suptitle(f"Effect of POD rank on the posterior (noise {mid_sigma:g}); black = full solver, orange = naive, blue dashed = discrepancy",
                 fontweight="bold", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_dir / "fig3_rank_effect_on_posterior.png", dpi=200)
    plt.close(fig)

    # F4: distance to reference
    if len(df_cmp):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sig_list = sorted(df_cmp["sigma"].unique())
        for k, (metric, ttl) in enumerate([("shift_mahal_ref", "Posterior-mean shift vs full solver\n(in units of the reference posterior sd)"),
                                            ("sd_ratio_angle", "Posterior sd ratio vs full solver (angle)\n<1 means overconfident")]):
            ax = axes[k]
            for kind in ("naive", "disc"):
                for ri_, s in enumerate(surs):
                    sub = df_cmp[(df_cmp.likelihood == kind) & (df_cmp.pod_rank == s["rank"])]
                    mval = sub.groupby("sigma")[metric].mean().reindex(sig_list)
                    ax.plot(range(len(sig_list)), mval.values, marker="o", ls=RANK_LS[ri_ % 3], color=COL[kind], lw=1.6)
            ax.set_xticks(range(len(sig_list)))
            ax.set_xticklabels([f"{s:g}" for s in sig_list])
            ax.set_xlabel("noise sd")
            ax.set_yscale("log")
            ax.set_title(ttl)
            if metric == "sd_ratio_angle":
                ax.axhline(1.0, color=COL["ref"], lw=0.8)
        handles = [Line2D([], [], color=COL["naive"], lw=1.8), Line2D([], [], color=COL["disc"], lw=1.8)] + \
                  [Line2D([], [], color=COL["muted"], ls=RANK_LS[i % 3], lw=1.6) for i in range(len(surs))]
        axes[1].legend(handles, ["naive", "discrepancy"] + [f"rank {s['rank']}" for s in surs], loc="best", fontsize=8)
        fig.suptitle("Agreement with the full-solver posterior (mean over reference cases)", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(fig_dir / "fig4_distance_to_full_solver_posterior.png", dpi=200)
        plt.close(fig)

    # F5: coverage vs sigma
    fig, axes = plt.subplots(1, len(surs), figsize=(3.6 * len(surs) + 0.6, 3.6), sharey=True, squeeze=False)
    band = 1.96 * np.sqrt(0.9 * 0.1 / n_cal)
    for si, s in enumerate(surs):
        ax = axes[0, si]
        ax.axhspan(0.9 - band, 0.9 + band, color=COL["grid"], zorder=0)
        ax.axhline(0.9, color=COL["muted"], lw=0.8)
        for kind in ("naive", "disc"):
            sub = agg[(agg.pod_rank == s["rank"]) & (agg.likelihood == kind)].sort_values("sigma")
            ax.plot(range(len(sub)), sub["joint_cov90"].values, "o-", color=COL[kind], lw=1.8, label=KIND_LABEL[kind])
        sg_ = sorted(cfg["sigmas_cal"])
        ax.set_xticks(range(len(sg_)))
        ax.set_xticklabels([f"{x:g}" for x in sg_], rotation=0)
        ax.set_xlabel("noise sd")
        ax.set_title(f"POD rank {s['rank']}")
        ax.set_ylim(0, 1.02)
        if si == 0:
            ax.set_ylabel("share of truths inside the 90% region")
            ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Calibration over {n_cal} prior-drawn truths (grey band: 95% binomial range around 0.9)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "fig5_calibration_coverage.png", dpi=200)
    plt.close(fig)

    # F6: PIT histograms at the smallest noise, middle rank
    sg_min = min(cfg["sigmas_cal"])
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharey=True)
    nb = 10 if n_cal >= 50 else 5
    exp = n_cal / nb
    lo_b, hi_b = exp - 1.96 * np.sqrt(exp * (1 - 1 / nb)), exp + 1.96 * np.sqrt(exp * (1 - 1 / nb))
    for r_, nm in enumerate(["amp", "angle"]):
        for c_, kind in enumerate(("naive", "disc")):
            ax = axes[r_, c_]
            sub = df_cal[(df_cal.sigma == sg_min) & (df_cal.pod_rank == surs[mid]["rank"]) & (df_cal.likelihood == kind)]
            ax.axhspan(lo_b, hi_b, color=COL["grid"], zorder=0)
            ax.hist(sub[f"{nm}_pit"], bins=np.linspace(0, 1, nb + 1), color=COL[kind], edgecolor="#fcfcfb", linewidth=1.5)
            ax.set_title(f"{PARAM_LABELS[r_]} - {KIND_LABEL[kind]}")
            if c_ == 0:
                ax.set_ylabel("count")
            if r_ == 1:
                ax.set_xlabel("posterior CDF at the true value")
    fig.suptitle(f"PIT histograms, noise {sg_min:g}, POD rank {surs[mid]['rank']} (flat = calibrated)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fig_dir / "fig6_pit_histograms.png", dpi=200)
    plt.close(fig)

    # F7: identifiability map
    sg_id = sorted(cfg["sigmas_cal"])[len(cfg["sigmas_cal"]) // 2]
    sub = df_cal[(df_cal.sigma == sg_id) & (df_cal.pod_rank == surs[mid]["rank"]) & (df_cal.likelihood == "disc")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, nm in enumerate(["amp", "angle"]):
        sc = axes[k].scatter(sub["amp_truth"], sub["angle_truth"], c=np.log10(sub[f"{nm}_sd"]), cmap="Blues", s=45,
                             edgecolors=COL["muted"], linewidths=0.5)
        fig.colorbar(sc, ax=axes[k], label=f"log10 posterior sd of {PARAM_LABELS[k]}")
        axes[k].set_xlabel("true logk_amp")
        axes[k].set_ylabel("true angle (deg)")
        axes[k].set_title(f"Where is {PARAM_LABELS[k].split(' ')[0]} identifiable?")
    fig.suptitle(f"Posterior width over the prior box (noise {sg_id:g}, rank {surs[mid]['rank']}, discrepancy likelihood)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fig_dir / "fig7_identifiability_map.png", dpi=200)
    plt.close(fig)

    # F8: error vs noise
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, nm in enumerate(["amp", "angle"]):
        ax = axes[k]
        sgs = sorted(cfg["sigmas_cal"])
        for kind in ("naive", "disc"):
            for si, s in enumerate(surs):
                sub = agg[(agg.pod_rank == s["rank"]) & (agg.likelihood == kind)].sort_values("sigma")
                ax.plot(sub["sigma"], sub[f"{nm}_abs_err_median"], marker="o", ls=RANK_LS[si % 3], color=COL[kind], lw=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("noise sd")
        ax.set_ylabel(f"median |posterior mean - truth| ({'' if nm == 'amp' else 'deg'})")
        ax.set_title(PARAM_LABELS[k])
    handles = [Line2D([], [], color=COL["naive"], lw=1.8), Line2D([], [], color=COL["disc"], lw=1.8)] + \
              [Line2D([], [], color=COL["muted"], ls=RANK_LS[i % 3], lw=1.6) for i in range(len(surs))]
    axes[1].legend(handles, ["naive", "discrepancy"] + [f"rank {s['rank']}" for s in surs], fontsize=8)
    fig.suptitle("Estimation error vs measurement noise: a floor means surrogate error, not noise, limits accuracy", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "fig8_error_vs_noise.png", dpi=200)
    plt.close(fig)

    # ---- tables / Excel --------------------------------------------------------------
    log("Writing Excel workbook and CSV tables...")
    df_diag = pd.concat([
        df_case[["case", "sigma", "pod_rank", "likelihood", "rhat_max", "ess_min", "acceptance", "n_modes"]].assign(source="case study"),
        df_cal[["sigma", "pod_rank", "likelihood", "rhat_max", "ess_min", "acceptance", "n_modes"]].assign(source="calibration"),
    ], ignore_index=True)
    readme = pd.DataFrame({"sheet": [
        "Config", "Self_tests", "Surrogate_quality", "Case_studies", "Reference_full_solver", "Reference_comparison",
        "Calibration_summary", "Calibration_cases", "Diagnostics_summary"], "contents": [
        "Settings used for this run", "Automatic correctness checks (all must be True)",
        "Surrogate accuracy at the sensors by POD rank (validation set)",
        "Posterior summaries for 3 truths x noise levels x surrogate arms",
        "Full-solver reference posteriors (MCMC) with diagnostics",
        "Distance of each surrogate posterior to the full-solver posterior",
        "Coverage / error / width per (noise, rank, likelihood) over prior truths",
        "One row per calibration case and arm", "MCMC health: share of runs with R-hat>1.05, ESS<200, multimodal flag"]})
    diag_sum = df_diag.groupby(["source", "pod_rank", "likelihood"]).agg(
        runs=("rhat_max", "count"), rhat_gt_1p05=("rhat_max", lambda r: float(np.mean(r > 1.05))),
        ess_lt_200=("ess_min", lambda e: float(np.mean(e < 200))), multimodal=("n_modes", lambda m: float(np.mean(m > 1))),
        acceptance_mean=("acceptance", "mean")).reset_index()
    cfg_df = pd.DataFrame({"setting": [k for k in cfg], "value": [str(v) for v in cfg.values()]})
    sheets = {
        "README": readme, "Config": cfg_df, "Self_tests": st, "Surrogate_quality": df_surr, "Case_studies": df_case,
        "Reference_full_solver": df_ref, "Reference_comparison": df_cmp, "Calibration_summary": agg,
        "Calibration_cases": df_cal, "Diagnostics_summary": diag_sum,
    }
    xlsx = out / "bayesian_inversion_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for nm, df in sheets.items():
            df.to_excel(xw, sheet_name=nm[:31], index=False)
            ws = xw.sheets[nm[:31]]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:60]) + 2
                ws.column_dimensions[col[0].column_letter].width = min(max(width, 8), 60)
            for c in ws[1]:
                c.font = c.font.copy(bold=True)
            ws.freeze_panes = "A2"
    for nm, df in sheets.items():
        if nm not in ("README", "Config"):
            df.to_csv(out / "tables" / f"{nm}.csv", index=False)

    # ---- summary.md -------------------------------------------------------------------
    elapsed = time.perf_counter() - T_start
    write_summary(out, cfg, df_surr, df_case, df_ref, df_cmp, agg, df_cal, diag_sum, st, surs, mid, elapsed)
    log(f"Done in {elapsed / 60:.1f} min. Results in {out.resolve()}")


def fmt(x: float, p: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}g}"


def write_summary(out: Path, cfg, df_surr, df_case, df_ref, df_cmp, agg, df_cal, diag_sum, st, surs, mid, elapsed) -> None:
    n_cal = cfg["n_cal"]
    band = 1.96 * np.sqrt(0.9 * 0.1 / n_cal)
    mr = surs[mid]["rank"]
    L: list[str] = []
    L.append("# Bayesian inversion with the POD-GPR surrogate: results summary\n")
    L.append(f"*Generated automatically in {elapsed / 60:.1f} min. Mode: {'QUICK (workflow test, not research results)' if cfg['quick'] else 'full'}. "
             f"Numbers below come straight from the workbook; wording is rule-based, so check it against the figures.*\n")
    L.append("## Setup\n")
    L.append(f"- Unknowns: logk_amp and pattern angle (uniform prior on the study box). logk_mean is not inferred because pressure does not depend on it.")
    L.append(f"- Data: pressure at {cfg['n_side']**2} sensors from the full {cfg['nx']}x{cfg['nx']} solver plus Gaussian noise (known sd).")
    L.append(f"- Surrogates: POD-GPR (RBF, same settings as Experiment A) with POD ranks {', '.join(str(s['rank']) for s in surs)}, trained on {cfg['n_train']} LHS solves.")
    L.append(f"- Two likelihoods: *naive* (surrogate treated as exact) and *discrepancy* (surrogate error modelled as Gaussian from {cfg['n_val']} validation cases).")
    L.append(f"- Self-tests passed: {int(st['passed'].sum())}/{len(st)}.\n")
    L.append("## Surrogate error at the sensors vs measurement noise\n")
    L.append("| POD rank | validation field rel. L2 | sensor RMS error | bias-removed RMS |\n|---|---|---|---|")
    for _, r in df_surr.iterrows():
        L.append(f"| {int(r.pod_rank)} | {fmt(r.validation_field_rel_L2)} | {fmt(r.sensor_residual_RMS)} | {fmt(r.sensor_residual_RMS_bias_removed)} |")
    L.append("")
    cross = [s for s in cfg["sigmas_cal"] if s < surs[mid]["rms_sensor_resid"]]
    if cross:
        L.append(f"For rank {mr}, noise levels below about {fmt(surs[mid]['rms_sensor_resid'])} are smaller than the surrogate's own error at the sensors "
                 f"({', '.join(f'{s:g}' for s in cross)} in this run). That is where a naive posterior is expected to be too sharp.\n")
    else:
        L.append(f"For rank {mr}, every noise level in this run is above the surrogate's sensor error, so large naive-vs-discrepancy differences are not expected.\n")

    L.append("## Calibration over prior-drawn truths\n")
    L.append(f"Share of {n_cal} truths inside the joint 90% posterior region (target 0.90, 95% binomial range {0.9 - band:.2f}-{min(1.0, 0.9 + band):.2f}; "
             f"the 90% region is judged by posterior-sample Mahalanobis distance).\n")
    head = "| noise | " + " | ".join(f"rank {s['rank']} naive | rank {s['rank']} disc." for s in surs) + " |"
    L.append(head)
    L.append("|---|" + "---|" * (2 * len(surs)))
    for sg in sorted(cfg["sigmas_cal"]):
        cells = []
        for s in surs:
            for kind in ("naive", "disc"):
                v = agg[(agg.sigma == sg) & (agg.pod_rank == s["rank"]) & (agg.likelihood == kind)]["joint_cov90"]
                cells.append(fmt(float(v.iloc[0]), 2))
        L.append(f"| {sg:g} | " + " | ".join(cells) + " |")
    L.append("")
    smin = min(cfg["sigmas_cal"])
    for s in surs:
        vn = float(agg[(agg.sigma == smin) & (agg.pod_rank == s["rank"]) & (agg.likelihood == "naive")]["joint_cov90"].iloc[0])
        vd = float(agg[(agg.sigma == smin) & (agg.pod_rank == s["rank"]) & (agg.likelihood == "disc")]["joint_cov90"].iloc[0])
        tag_n = "within" if abs(vn - 0.9) <= band else ("below: overconfident, outside" if vn < 0.9 else "above: conservative, outside")
        tag_d = "within" if abs(vd - 0.9) <= band else ("below: overconfident, outside" if vd < 0.9 else "above: conservative, outside")
        L.append(f"- Rank {s['rank']}, noise {smin:g}: naive {vn:.2f} ({tag_n} the binomial range), discrepancy {vd:.2f} ({tag_d}).")
    L.append("")

    L.append("## Case studies vs the full-solver posterior\n")
    if len(df_cmp):
        sub = df_cmp[df_cmp.pod_rank == mr]
        L.append(f"Rank {mr}. *shift* = distance between posterior means in units of the reference posterior sd; *sd ratio* = surrogate posterior sd / reference sd (angle).\n")
        L.append("| case | noise | naive shift | disc. shift | naive sd ratio | disc. sd ratio |\n|---|---|---|---|---|---|")
        for (cn, sg), g in sub.groupby(["case", "sigma"]):
            n_ = g[g.likelihood == "naive"].iloc[0]
            d_ = g[g.likelihood == "disc"].iloc[0]
            L.append(f"| {cn} | {sg:g} | {fmt(n_.shift_mahal_ref)} | {fmt(d_.shift_mahal_ref)} | {fmt(n_.sd_ratio_angle)} | {fmt(d_.sd_ratio_angle)} |")
        L.append("")
        L.append("Reference chain health:\n")
        L.append("| case | noise | acceptance | max R-hat | min ESS | modes |\n|---|---|---|---|---|---|")
        for _, r in df_ref.iterrows():
            L.append(f"| {r.case} | {r.sigma:g} | {fmt(r.acceptance)} | {fmt(r.rhat_max, 3)} | {fmt(r.ess_min, 3)} | {int(r.n_modes)} |")
        L.append("")
    else:
        L.append("Full-solver reference posteriors were skipped in this run.\n")

    L.append("## MCMC health (all surrogate runs)\n")
    L.append("| source | rank | likelihood | runs | R-hat > 1.05 | ESS < 200 | multimodal flag |\n|---|---|---|---|---|---|---|")
    for _, r in diag_sum.iterrows():
        L.append(f"| {r.source} | {int(r.pod_rank)} | {r.likelihood} | {int(r.runs)} | {fmt(r.rhat_gt_1p05)} | {fmt(r.ess_lt_200)} | {fmt(r.multimodal)} |")
    L.append("")
    L.append("## What this does and does not show\n")
    L.append("- It tests the *surrogate* in an inference task with synthetic data from the same solver. It says nothing about real measurements, model misfit of the PDE itself, or non-Gaussian noise.")
    L.append("- The discrepancy model is stationary (one covariance for the whole box). Surrogate error is larger in some regions, so local miscalibration can remain.")
    L.append("- Calibration uses truths drawn from the prior; behaviour for a specific truth (see the case studies) can differ.")
    L.append("- Multimodality is flagged by multi-start optimisation on the grid, not proved absent. A multimodal flag with high R-hat means the chains did not mix between modes.")
    L.append("- Reference chains are short; trust them only where R-hat is close to 1 and ESS is in the hundreds.")
    L.append("- The surrogate is a single training set (one seed); no variation over training designs is estimated.\n")
    L.append("## Files\n")
    L.append("- `bayesian_inversion_results.xlsx` (all tables), `tables/*.csv`, `figures/fig1...fig8*.png`.")
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
