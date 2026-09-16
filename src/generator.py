"""
Matched-counterfactual data generator for the regime-aware causal discovery
benchmark.

Generates three scenario types that share the same underlying DAG and the
same randomly-selected set of "affected" nodes, calibrated to produce a
comparably sized marginal variance increase at a changepoint restricted to
those affected nodes -- but differ in *why* the shift happens:

- "mechanism":   the causal graph itself changes at the changepoint: each
                 affected node gains/strengthens a self-loop (its own lag-1
                 persistence increases) -- a TRUE mechanism change. This is
                 a provably variance-increasing perturbation in isolation
                 (Var of an AR(1) component = q / (1 - phi^2), monotonically
                 increasing in phi), and is verified at runtime.
- "confounding": the graph is unchanged; an unobserved common cause (loaded
                 only onto the affected nodes) increases its variance at the
                 changepoint, inflating the observed variance of exactly the
                 affected nodes without any true structural change among
                 observed variables.
- "noise":       the graph is unchanged; only the idiosyncratic noise
                 variance of the affected nodes increases at the changepoint.

Data model: a linear-Gaussian VAR(1) process with an additional unobserved
scalar confounder,

    X_t = A @ X_{t-1} + b * U_t + eps_t
    U_t ~ N(0, sigma_u^2),  eps_t ~ N(0, diag(sigma_eps^2))

where A is the lag-1 weighted adjacency matrix of the DAG among observed
nodes ("j -> i" iff A[i, j] != 0), and b is a fixed loading vector picking
out which nodes the latent confounder affects.

Shared randomization: for a given seed, a single "affected_frac ~
Uniform(0.25, 1.0)" draw and a single random subset of that many nodes
(`affected_nodes`) are drawn *before* branching on scenario_type, and reused
by all three scenarios: it decides which nodes the confounder loads onto,
which nodes' noise increases, and which nodes get a strengthened self-loop.
So for a fixed seed, all three scenarios target the *same* nodes.

Calibration: for a stable VAR(1) process, the stationary covariance Sigma
solves the discrete Lyapunov equation Sigma = A Sigma A^T + Q, with
Q = b b^T sigma_u^2 + diag(sigma_eps^2). Each scenario's post-changepoint
perturbation strength is calibrated via 1-D root finding so the resulting
increase in *average marginal variance of the affected nodes only* (mean of
diag(Sigma_post) over affected_nodes minus the same over Sigma_pre) matches
the same target across all three scenarios -- i.e. the size and the set of
nodes carrying the marginal shift are matched by construction, only the
cause differs.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import networkx as nx
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import brentq

SCENARIO_TYPES = ("mechanism", "confounding", "noise")


@dataclasses.dataclass
class MatchedScenario:
    """Output of generate_matched_scenario."""

    data: np.ndarray  # (T, n_nodes) observed time series
    dag_pre: nx.DiGraph  # true causal graph among observed nodes, pre-changepoint
    dag_post: nx.DiGraph  # true causal graph among observed nodes, post-changepoint
    changepoint: int  # index t* where the regime switches (post regime starts at t*)
    scenario_type: str
    n_nodes: int
    T: int
    seed: int
    affected_nodes: np.ndarray  # node indices targeted by the changepoint perturbation
    shift_size: float  # achieved increase in avg marginal variance of affected_nodes (post - pre)
    latent_confounder: Optional[np.ndarray] = None  # (T,) values of U_t, diagnostics only


def _weighted_var1_matrix(n_nodes, edge_prob, rng, spectral_radius=0.6):
    """Random sparse weighted VAR(1) matrix A; A[i, j] != 0 means j -> i at lag 1.

    Diagonal is left at zero here -- self-loops are reserved for the
    "mechanism" scenario's changepoint perturbation.
    """
    A = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            if rng.random() < edge_prob:
                A[i, j] = rng.uniform(0.3, 0.8) * rng.choice([-1.0, 1.0])
    rho = np.max(np.abs(np.linalg.eigvals(A))) if np.any(A) else 0.0
    if rho > 0:
        A *= spectral_radius / rho
    return A


def _draw_affected_nodes(n_nodes, rng, frac_low=0.25, frac_high=1.0):
    """Shared randomization: fraction and identity of nodes targeted by the changepoint,
    drawn once per seed and reused by all three scenario types.
    """
    affected_frac = rng.uniform(frac_low, frac_high)
    n_affected = max(1, int(round(affected_frac * n_nodes)))
    affected_nodes = np.sort(rng.choice(n_nodes, size=n_affected, replace=False))
    return affected_nodes


def _confounder_loading(n_nodes, affected_nodes, rng):
    """Loading vector for the unobserved confounder: nonzero only on affected_nodes."""
    b = np.zeros(n_nodes)
    b[affected_nodes] = rng.uniform(0.5, 1.0, size=len(affected_nodes)) * rng.choice(
        [-1.0, 1.0], size=len(affected_nodes)
    )
    return b


def _safe_stationary_cov(A, Q):
    """Stationary covariance Sigma = A Sigma A^T + Q, or None if A is not stable."""
    try:
        if np.max(np.abs(np.linalg.eigvals(A))) >= 0.995:
            return None
        return solve_discrete_lyapunov(A, Q)
    except np.linalg.LinAlgError:
        return None


def _avg_var_on(Sigma, nodes):
    return float(np.mean(np.diag(Sigma)[nodes]))


def _graph_from_matrix(A, threshold=1e-8):
    """Build a networkx DiGraph (edge j -> i, lag 1) from a VAR(1) weight matrix."""
    n = A.shape[0]
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if abs(A[i, j]) > threshold:
                g.add_edge(j, i, weight=float(A[i, j]), lag=1)
    return g


def _calibrate_strength(model_fn, affected_nodes, base_avg_var, target_shift, s_max=20.0):
    """Find scale s >= 0 such that avg_var_on(model_fn(s), affected_nodes) - base_avg_var == target_shift.

    model_fn(s) -> (A_s, Q_s). Stays within the region where A_s is stable;
    treats instability as "overshoot" so the search brackets below it.
    """

    def f(s):
        A_s, Q_s = model_fn(s)
        Sigma = _safe_stationary_cov(A_s, Q_s)
        if Sigma is None:
            return 1e6
        return _avg_var_on(Sigma, affected_nodes) - base_avg_var - target_shift

    lo, hi = 0.0, s_max
    if f(lo) >= 0:
        return lo
    f_hi = f(hi)
    tries = 0
    while f_hi < 0 and tries < 8:
        hi *= 1.5
        f_hi = f(hi)
        tries += 1
    if f_hi < 0:
        return hi  # best effort: never reached target within a stable range
    return brentq(f, lo, hi, xtol=1e-6)


def generate_matched_scenario(
    n_nodes: int,
    T: int,
    seed: int,
    scenario_type: str,
    changepoint: Optional[int] = None,
    edge_prob: float = 0.35,
    spectral_radius: float = 0.6,
    base_noise_std: float = 1.0,
    base_confounder_std: float = 1.0,
    affected_frac_range: tuple = (0.25, 1.0),
    target_shift_ratio: float = 0.6,
    burn_in: int = 50,
) -> MatchedScenario:
    """Generate a single matched-counterfactual scenario.

    All three scenario types share the same base DAG (`A_base`), the same
    randomly-drawn set of `affected_nodes` (see module docstring), the same
    latent-confounder loading vector, and the same base noise levels. They
    differ only in *what* changes at the changepoint for the affected nodes
    (self-loop strength / confounder variance / idiosyncratic noise
    variance), with the change strength calibrated so the resulting shift in
    average marginal variance *of the affected nodes* matches
    `target_shift_ratio * base average marginal variance of affected nodes`
    in all three cases.

    :param n_nodes: number of observed nodes
    :param T: number of time steps in the returned series
    :param seed: random seed
    :param scenario_type: one of "mechanism", "confounding", "noise"
    :param changepoint: index where the regime switches; defaults to T // 2
    :param edge_prob: probability of a lagged edge between any ordered pair of nodes
    :param spectral_radius: target spectral radius of the base VAR(1) matrix (stability)
    :param base_noise_std: pre-changepoint idiosyncratic noise std (per node)
    :param base_confounder_std: pre-changepoint latent confounder std
    :param affected_frac_range: (low, high) range to draw the fraction of
        nodes targeted by the changepoint perturbation from, uniformly
    :param target_shift_ratio: target relative increase in average marginal
        variance of the affected nodes at the changepoint (e.g. 0.6 = +60%),
        matched across scenarios
    :param burn_in: samples simulated and discarded before t=0, so the
        pre-changepoint regime starts near its stationary distribution
    :return: MatchedScenario
    """
    if scenario_type not in SCENARIO_TYPES:
        raise ValueError(f"scenario_type must be one of {SCENARIO_TYPES}, got {scenario_type!r}")
    if changepoint is None:
        changepoint = T // 2
    if not (0 < changepoint < T):
        raise ValueError("changepoint must be strictly between 0 and T")

    rng = np.random.default_rng(seed)

    # --- shared base model + shared node randomization (identical across all scenario types) ---
    A_base = _weighted_var1_matrix(n_nodes, edge_prob, rng, spectral_radius)
    affected_nodes = _draw_affected_nodes(n_nodes, rng, *affected_frac_range)
    b = _confounder_loading(n_nodes, affected_nodes, rng)
    sigma_eps_base = np.full(n_nodes, base_noise_std)
    sigma_u_base = base_confounder_std

    Q_base = np.outer(b, b) * sigma_u_base**2 + np.diag(sigma_eps_base**2)
    Sigma_pre = _safe_stationary_cov(A_base, Q_base)
    base_avg_var = _avg_var_on(Sigma_pre, affected_nodes)
    target_shift = target_shift_ratio * base_avg_var

    if scenario_type == "mechanism":
        # strengthen each affected node's own lag-1 self-loop (phi_i: 0 -> s).
        # Var(AR(1) component) = q / (1 - phi^2) is monotonically increasing in
        # |phi| for phi in (-1, 1), so this provably increases the affected
        # node's variance in isolation; verified against the full multivariate
        # Lyapunov solution below.
        def model_fn(s):
            A_s = A_base.copy()
            for i in affected_nodes:
                A_s[i, i] = s
            return A_s, Q_base

        s = _calibrate_strength(model_fn, affected_nodes, base_avg_var, target_shift)
        A_post, Q_post = model_fn(s)
        sigma_u_post, sigma_eps_post = sigma_u_base, sigma_eps_base
    elif scenario_type == "confounding":
        A_post = A_base

        def model_fn(s):
            return A_base, np.outer(b, b) * (sigma_u_base + s) ** 2 + np.diag(sigma_eps_base**2)

        s = _calibrate_strength(model_fn, affected_nodes, base_avg_var, target_shift)
        A_post, Q_post = model_fn(s)
        sigma_u_post, sigma_eps_post = sigma_u_base + s, sigma_eps_base
    else:  # noise
        def model_fn(s):
            sigma_eps_s = sigma_eps_base.copy()
            sigma_eps_s[affected_nodes] += s
            return A_base, np.outer(b, b) * sigma_u_base**2 + np.diag(sigma_eps_s**2)

        s = _calibrate_strength(model_fn, affected_nodes, base_avg_var, target_shift)
        A_post, Q_post = model_fn(s)
        sigma_eps_post = sigma_eps_base.copy()
        sigma_eps_post[affected_nodes] += s
        sigma_u_post = sigma_u_base

    Sigma_post = _safe_stationary_cov(A_post, Q_post)
    post_avg_var = _avg_var_on(Sigma_post, affected_nodes)
    achieved_shift = post_avg_var - base_avg_var

    # runtime guarantee: every affected node's own marginal variance must have
    # increased (not just the average across affected nodes)
    pre_diag = np.diag(Sigma_pre)
    post_diag = np.diag(Sigma_post)
    if np.any(post_diag[affected_nodes] <= pre_diag[affected_nodes]):
        raise RuntimeError(
            f"scenario_type={scenario_type!r}, seed={seed}: calibration failed to increase "
            f"variance for every affected node individually "
            f"(pre={pre_diag[affected_nodes]}, post={post_diag[affected_nodes]})"
        )

    # --- simulate the actual finite time series (with burn-in for the pre-regime) ---
    T_total = burn_in + T
    X = np.zeros((T_total, n_nodes))
    U = np.zeros(T_total)
    x_prev = np.zeros(n_nodes)
    for t in range(T_total):
        t_data = t - burn_in  # index on the "public" 0..T-1 timeline (negative during burn-in)
        is_post = t_data >= changepoint
        A_t = A_post if is_post else A_base
        sigma_u_t = sigma_u_post if is_post else sigma_u_base
        sigma_eps_t = sigma_eps_post if is_post else sigma_eps_base

        u_t = rng.normal(0.0, sigma_u_t)
        eps_t = rng.normal(0.0, sigma_eps_t)
        x_t = A_t @ x_prev + b * u_t + eps_t

        X[t] = x_t
        U[t] = u_t
        x_prev = x_t

    data = X[burn_in:]
    latent = U[burn_in:]

    return MatchedScenario(
        data=data,
        dag_pre=_graph_from_matrix(A_base),
        dag_post=_graph_from_matrix(A_post),
        changepoint=changepoint,
        scenario_type=scenario_type,
        n_nodes=n_nodes,
        T=T,
        seed=seed,
        affected_nodes=affected_nodes,
        shift_size=achieved_shift,
        latent_confounder=latent,
    )
