"""
Matched-counterfactual data generator for the regime-aware causal discovery
benchmark, calibrated via the exact closed-form MMD^2 result (linear-Gaussian
special case of Theorem 1, as supplied by the thesis supervisor).

SCOPE DECISION -- read before touching this file
==================================================
The closed-form calibration below (`mmd2_mechanism`, `mmd2_confound_or_noise`,
`delta_star`) is only *proven* for a single-parent structural equation

    Y = a * W + N,   W ~ N(0, sigma_W2),   N ~ N(0, sigma2)

i.e. exactly one parent W per tested child Y. This is a deliberate scope
restriction, not a limitation we ran into later: we do NOT attempt any
multivariate-MMD generalization to nodes with multiple parents, because that
is not what the theorem covers. Concretely:

- Every node whose changepoint behaviour we calibrate and measure (every
  "tested edge") has exactly one parent, by construction of the graph below.
- A node MAY have additional untested structure around it -- e.g. it may
  itself be the parent of other tested edges, or the graph may contain
  further source (parentless) nodes for decoration/realism -- but no tested
  edge is ever given a second parent.

Because the theorem's structural equation is a memoryless (non-temporal)
Gaussian SEM, the "time series" produced here is i.i.d. across time within
each regime (no autoregressive/lag component) -- the changepoint is a
regime switch in the SEM's parameters, not a dynamical transient. This
matches the theorem's assumptions exactly, rather than approximating them.

Model per tested edge, per scenario type
-----------------------------------------
- "mechanism":   a change to a' with |a - a'| = Delta_a. Distribution of the
                 *joint* (W, Y) changes; calibrated via `mmd2_mechanism`.
- "confounding": a is unchanged; an independent latent U with std `delta` is
                 added to Y (Y += delta-scaled U), inflating Y's marginal
                 variance by delta^2 without changing the tested edge's
                 structural coefficient.
- "noise":       a is unchanged; N's variance increases from sigma2 to
                 sigma2 + delta^2 directly.

For a fixed sigma2 and gamma, `mmd2_confound_or_noise(delta, sigma2, gamma)`
is identical in form for confounding and noise (both are "Y's total
variance goes from sigma2 to sigma2 + delta^2"); they differ only in *how*
that extra variance enters the DGP (a shared latent vs. inflated residual
noise), which is what a benchmark method would have to exploit -- the
tested edge's own marginal distribution is, by design, indistinguishable
between the two.

Calibration: given a target mechanism change Delta_a (expressed as a
fraction of `delta_star`, the exact threshold at which
`mmd2_mechanism(Delta_a, ...)` equals the supremum of
`mmd2_confound_or_noise` as delta -> infinity), we solve for the `delta`
that makes `mmd2_confound_or_noise(delta, ...)` match `mmd2_mechanism`
exactly, via 1-D root finding (the theorem guarantees monotonicity in
delta^2, hence a unique root when a match is possible).

Below threshold (Delta_a = 0.5 * delta_star): a matching delta is
guaranteed to exist (Satz A) -- calibration must succeed.
Above threshold (Delta_a = 2.0 * delta_star): no finite delta can match
(Satz B) -- calibration must fail, and does so explicitly via
`CalibrationImpossibleError` rather than silently returning a mismatched
scenario.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import networkx as nx
import numpy as np
from scipy.optimize import brentq

SCENARIO_TYPES = ("mechanism", "confounding", "noise")
THRESHOLD_REGIMES = ("below_threshold", "above_threshold")
_REGIME_FRACTION = {"below_threshold": 0.5, "above_threshold": 2.0}


class CalibrationImpossibleError(RuntimeError):
    """Raised when no finite confound/noise delta can match a mechanism shift
    (i.e. the requested Delta_a is at or above delta_star -- Satz B)."""


# --------------------------------------------------------------------------
# Step 1: exact closed-form formulas (linear-Gaussian special case of Theorem 1)
# --------------------------------------------------------------------------

def c(var, gamma):
    return gamma / math.sqrt(gamma**2 + 2 * var)


def mmd2_mechanism(delta_a, sigma2, gamma, sigma_W2):
    kappa0 = delta_a**2 / (2 * (gamma**2 + 2 * sigma2))
    return 2 * c(sigma2, gamma) * (1 - 1 / math.sqrt(1 + 2 * kappa0 * sigma_W2))


def mmd2_confound_or_noise(delta, sigma2, gamma):
    return (
        c(sigma2, gamma)
        + c(sigma2 + delta**2, gamma)
        - 2 * gamma / math.sqrt(gamma**2 + 2 * sigma2 + delta**2)
    )


def delta_star(sigma2, gamma, sigma_W2):
    return math.sqrt(3 * (gamma**2 + 2 * sigma2) / sigma_W2)


# --------------------------------------------------------------------------
# Step 2: MMD^2-based calibration (replaces the old variance-ratio calibration)
# --------------------------------------------------------------------------

def solve_matching_delta(target_mmd2, sigma2, gamma, delta_max=1e6, xtol=1e-12):
    """Solve for delta >= 0 such that mmd2_confound_or_noise(delta, sigma2, gamma) == target_mmd2.

    mmd2_confound_or_noise is 0 at delta=0 and increases monotonically in
    delta^2 towards the supremum c(sigma2, gamma) as delta -> infinity
    (Theorem 1). If target_mmd2 is at or beyond that supremum, no finite
    delta can match it -- raises CalibrationImpossibleError explicitly
    instead of returning a bogus/mismatched value.
    """
    sup = c(sigma2, gamma)
    if target_mmd2 >= sup - 1e-12:
        raise CalibrationImpossibleError(
            f"target MMD^2={target_mmd2:.6g} >= supremum c(sigma2,gamma)={sup:.6g}: "
            "no finite confound/noise delta can match this mechanism shift (Satz B)."
        )

    def f(delta):
        return mmd2_confound_or_noise(delta, sigma2, gamma) - target_mmd2

    lo, hi = 0.0, 1.0
    tries = 0
    while f(hi) < 0:
        hi *= 2.0
        tries += 1
        if hi > delta_max or tries > 200:
            raise CalibrationImpossibleError(
                f"target MMD^2={target_mmd2:.6g} not reached within delta_max={delta_max:.3g} "
                "(supremum check passed but the search did not bracket a root -- investigate)."
            )
    return brentq(f, lo, hi, xtol=xtol)


# --------------------------------------------------------------------------
# Graph construction: every tested edge has exactly one parent (scope decision above)
# --------------------------------------------------------------------------

@dataclasses.dataclass
class TestedEdge:
    """Diagnostic record for one calibrated single-parent edge."""

    child: int
    parent: int
    sigma2: float
    sigma_W2: float
    gamma: float
    delta_star: float
    delta_a_target: Optional[float] = None  # mechanism: |a - a'| applied
    delta_confound_or_noise: Optional[float] = None  # confounding/noise: solved delta
    target_mmd2: float = 0.0
    achieved_mmd2: float = 0.0


@dataclasses.dataclass
class MatchedScenario:
    """Output of generate_matched_scenario."""

    data: np.ndarray  # (T, n_nodes) observed time series (i.i.d. across t within each regime)
    dag_pre: nx.DiGraph
    dag_post: nx.DiGraph
    changepoint: int
    scenario_type: str
    strength_regime: str  # "below_threshold" or "above_threshold"
    n_nodes: int
    T: int
    seed: int
    affected_nodes: np.ndarray  # child nodes whose tested edge was perturbed at the changepoint
    tested_edges: list  # list[TestedEdge], diagnostics for each perturbed edge
    latent_confounder: Optional[np.ndarray] = None  # (T, n_affected) confound draws, diagnostics only


def _draw_affected_nodes(n_nodes, rng, frac_low=0.25, frac_high=1.0):
    """Shared randomization: which child nodes get a tested (single-parent) edge
    perturbed at the changepoint. Node 0 is always a pure source node so every
    other node has an earlier node available to serve as its single parent.
    """
    candidates = np.arange(1, n_nodes)
    affected_frac = rng.uniform(frac_low, frac_high)
    n_affected = max(1, int(round(affected_frac * len(candidates))))
    affected_nodes = np.sort(rng.choice(candidates, size=n_affected, replace=False))
    return affected_nodes


def _build_base_sem(n_nodes, affected_nodes, rng, base_noise_std):
    """Builds a feedforward (topological order 0..n-1), single-parent-per-tested-edge
    linear-Gaussian SEM shared by all three scenario types.

    Returns:
        parent_of: dict child -> parent (only for affected_nodes)
        a_pre: dict child -> pre-changepoint coefficient
        sigma2: float, shared residual/noise variance (same for every node)
        var_of: dict node -> marginal variance (Var[node]), computed analytically
                 in topological order -- this is exact because the SEM is a
                 feedforward sum of independent zero-mean Gaussians.
    """
    sigma2 = base_noise_std**2
    parent_of = {}
    a_pre = {}
    var_of = {}

    affected_set = set(int(i) for i in affected_nodes)
    for i in range(n_nodes):
        if i in affected_set:
            parent = int(rng.integers(0, i))  # any earlier node, uniformly
            a = rng.uniform(0.3, 0.8) * rng.choice([-1.0, 1.0])
            parent_of[i] = parent
            a_pre[i] = a
            var_of[i] = a**2 * var_of[parent] + sigma2
        else:
            var_of[i] = sigma2  # pure source node

    return parent_of, a_pre, sigma2, var_of


def _graph_from_parents(n_nodes, parent_of, a_map):
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    for child, parent in parent_of.items():
        g.add_edge(parent, child, weight=float(a_map[child]), lag=0)
    return g


# --------------------------------------------------------------------------
# Step 3/4: scenario generation using the MMD^2 calibration exclusively
# --------------------------------------------------------------------------

def generate_matched_scenario(
    n_nodes: int,
    T: int,
    seed: int,
    scenario_type: str,
    strength_regime: str = "below_threshold",
    changepoint: Optional[int] = None,
    gamma: float = 1.0,
    base_noise_std: float = 1.0,
    affected_frac_range: tuple = (0.25, 1.0),
    burn_in: int = 0,
) -> MatchedScenario:
    """Generate a single matched-counterfactual scenario, calibrated via the
    exact MMD^2 closed form (see module docstring for the scope decision and
    the model). No autoregressive dynamics: data is i.i.d. across time within
    each regime (a deliberate consequence of the theorem's static SEM scope).

    :param n_nodes: number of observed nodes (node 0 is always a pure source node)
    :param T: number of time steps in the returned series
    :param seed: random seed; all three scenario types share the same base
        SEM (topology, coefficients, affected_nodes) for a given seed
    :param scenario_type: one of "mechanism", "confounding", "noise"
    :param strength_regime: "below_threshold" (Delta_a = 0.5 * delta_star,
        calibration must succeed) or "above_threshold" (Delta_a = 2.0 *
        delta_star, calibration must fail for confounding/noise -- raises
        CalibrationImpossibleError)
    :param changepoint: index where the regime switches; defaults to T // 2
    :param gamma: Gaussian-kernel bandwidth, shared by every tested edge
        (a fixed hyperparameter of the generator, not fit to data)
    :param base_noise_std: residual/noise std, shared by every node
    :param affected_frac_range: (low, high) range to draw the fraction of
        (non-source) nodes that get a tested edge from, uniformly
    :param burn_in: kept for interface compatibility; unused since the SEM
        is memoryless (i.i.d. across t), there is no transient to burn in
    :return: MatchedScenario
    :raises CalibrationImpossibleError: for scenario_type in {"confounding",
        "noise"} when strength_regime="above_threshold" (Satz B)
    """
    if scenario_type not in SCENARIO_TYPES:
        raise ValueError(f"scenario_type must be one of {SCENARIO_TYPES}, got {scenario_type!r}")
    if strength_regime not in THRESHOLD_REGIMES:
        raise ValueError(f"strength_regime must be one of {THRESHOLD_REGIMES}, got {strength_regime!r}")
    if changepoint is None:
        changepoint = T // 2
    if not (0 < changepoint < T):
        raise ValueError("changepoint must be strictly between 0 and T")

    rng = np.random.default_rng(seed)

    # --- shared base SEM + shared node randomization (identical across all scenario types) ---
    affected_nodes = _draw_affected_nodes(n_nodes, rng, *affected_frac_range)
    parent_of, a_pre, sigma2, var_of = _build_base_sem(n_nodes, affected_nodes, rng, base_noise_std)

    fraction = _REGIME_FRACTION[strength_regime]

    a_post = dict(a_pre)
    sigma2_post = {i: sigma2 for i in affected_nodes}
    confound_delta = {i: 0.0 for i in affected_nodes}
    tested_edges = []

    for i in affected_nodes:
        parent = parent_of[i]
        sigma_W2 = var_of[parent]
        d_star = delta_star(sigma2, gamma, sigma_W2)
        delta_a_target = fraction * d_star
        target_mmd2 = mmd2_mechanism(delta_a_target, sigma2, gamma, sigma_W2)

        edge = TestedEdge(
            child=int(i), parent=int(parent), sigma2=sigma2, sigma_W2=sigma_W2,
            gamma=gamma, delta_star=d_star,
        )

        if scenario_type == "mechanism":
            # provable variance-increase constraint: grow |a| by delta_a_target
            a0 = a_pre[i]
            a1 = math.copysign(abs(a0) + delta_a_target, a0) if a0 != 0 else delta_a_target
            a_post[i] = a1
            edge.delta_a_target = delta_a_target
            edge.target_mmd2 = target_mmd2
            edge.achieved_mmd2 = mmd2_mechanism(abs(a1 - a0), sigma2, gamma, sigma_W2)
        else:  # confounding or noise: solve for the matching delta (may raise)
            delta = solve_matching_delta(target_mmd2, sigma2, gamma)
            edge.delta_confound_or_noise = delta
            edge.target_mmd2 = target_mmd2
            edge.achieved_mmd2 = mmd2_confound_or_noise(delta, sigma2, gamma)
            if scenario_type == "confounding":
                confound_delta[i] = delta
            else:  # noise
                sigma2_post[i] = sigma2 + delta**2

        tested_edges.append(edge)

    # runtime guarantee: every tested edge's post-changepoint variance strictly increases
    for edge in tested_edges:
        i = edge.child
        var_pre = a_pre[i] ** 2 * var_of[parent_of[i]] + sigma2
        if scenario_type == "mechanism":
            var_post = a_post[i] ** 2 * var_of[parent_of[i]] + sigma2
        elif scenario_type == "confounding":
            var_post = var_pre + confound_delta[i] ** 2
        else:
            var_post = a_pre[i] ** 2 * var_of[parent_of[i]] + sigma2_post[i]
        if not (var_post > var_pre):
            raise RuntimeError(
                f"scenario_type={scenario_type!r}, seed={seed}, child={i}: "
                f"post-changepoint variance did not increase (pre={var_pre}, post={var_post})"
            )

    # --- simulate the actual finite time series (i.i.d. across t within each regime) ---
    affected_list = list(affected_nodes)
    X = np.zeros((T, n_nodes))
    U = np.zeros((T, len(affected_list)))
    for t in range(T):
        is_post = t >= changepoint
        for i in range(n_nodes):
            if i not in parent_of:
                X[t, i] = rng.normal(0.0, math.sqrt(sigma2))
                continue
            p = parent_of[i]
            a = a_post[i] if (is_post and scenario_type == "mechanism") else a_pre[i]
            s2 = sigma2_post[i] if (is_post and scenario_type == "noise") else sigma2
            y = a * X[t, p] + rng.normal(0.0, math.sqrt(s2))
            if scenario_type == "confounding" and is_post:
                idx = affected_list.index(i)
                u = rng.normal(0.0, confound_delta[i])
                U[t, idx] = u
                y += u
            X[t, i] = y

    dag_pre = _graph_from_parents(n_nodes, parent_of, a_pre)
    dag_post = _graph_from_parents(n_nodes, parent_of, a_post if scenario_type == "mechanism" else a_pre)

    return MatchedScenario(
        data=X,
        dag_pre=dag_pre,
        dag_post=dag_post,
        changepoint=changepoint,
        scenario_type=scenario_type,
        strength_regime=strength_regime,
        n_nodes=n_nodes,
        T=T,
        seed=seed,
        affected_nodes=affected_nodes,
        tested_edges=tested_edges,
        latent_confounder=U if scenario_type == "confounding" else None,
    )
