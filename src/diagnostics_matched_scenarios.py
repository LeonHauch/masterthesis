"""
Phase 1 diagnostics for the matched-counterfactual generator (src/generator.py).

Prints plain-number checks for a fixed seed, without changing any generator
logic:

1. PARAMETER CHECK -- the actual VAR(1) edge-weight matrix (dag_pre vs.
   dag_post) for each scenario, so mechanism changes (or the lack thereof)
   can be verified by eye.
2. VISIBILITY CHECK -- shape/columns of the data array as it would be handed
   to a causal discovery method, confirming the latent confounder is not
   one of its columns.
3. VARIANCE-SHIFT CHECK -- per-variable (not averaged) empirical variance in
   a window before vs. after the changepoint, for each scenario.

Run: python src/diagnostics_matched_scenarios.py
"""
import numpy as np
import networkx as nx

from generator import SCENARIO_TYPES, generate_matched_scenario

N_NODES = 4
T = 200
SEED = 0
WINDOW = 50  # samples immediately before / after the changepoint


def graph_to_matrix(g: nx.DiGraph, n: int) -> np.ndarray:
    """A[i, j] = weight of edge j -> i (matches generator._graph_from_matrix)."""
    A = np.zeros((n, n))
    for j, i, data in g.edges(data=True):
        A[i, j] = data["weight"]
    return A


def print_matrix(A: np.ndarray, label: str):
    print(f"  {label}:")
    for row in A:
        print("    [" + ", ".join(f"{v:+.4f}" for v in row) + "]")


def main():
    scenarios = {
        st: generate_matched_scenario(n_nodes=N_NODES, T=T, seed=SEED, scenario_type=st)
        for st in SCENARIO_TYPES
    }

    print("=" * 70)
    print("1. PARAMETER CHECK -- VAR(1) edge-weight matrix A (A[i,j] = j -> i)")
    print("=" * 70)
    for st, sc in scenarios.items():
        A_pre = graph_to_matrix(sc.dag_pre, N_NODES)
        A_post = graph_to_matrix(sc.dag_post, N_NODES)
        identical = np.allclose(A_pre, A_post)
        print(f"\n[{st}]  A_pre == A_post ? {identical}")
        print_matrix(A_pre, "A_pre  (t < 100)")
        print_matrix(A_post, "A_post (t >= 100)")
        if not identical:
            diff = A_post - A_pre
            changed = np.argwhere(np.abs(diff) > 1e-8)
            print("  Changed entries (i <- j : pre -> post):")
            for i, j in changed:
                print(f"    {i} <- {j} : {A_pre[i, j]:+.4f} -> {A_post[i, j]:+.4f}")

    print()
    print("=" * 70)
    print("2. VISIBILITY CHECK -- data handed to a causal discovery method")
    print("=" * 70)
    sc = scenarios["confounding"]
    columns = [f"X{i}" for i in range(N_NODES)]
    print(f"  scenario: confounding")
    print(f"  data.shape           = {sc.data.shape}   (T={sc.T}, n_nodes={N_NODES})")
    print(f"  data columns         = {columns}")
    print(f"  'Z' / confounder in data.columns? {'Z' in columns}")
    print(f"  latent_confounder is a SEPARATE array, shape = {sc.latent_confounder.shape}, "
          f"not concatenated into data")
    assert sc.data.shape[1] == N_NODES, "data must only contain the observed nodes"

    print()
    print("=" * 70)
    print("3. VARIANCE-SHIFT CHECK -- per-variable variance, before vs. after t=100")
    print(f"   (window = {WINDOW} samples on each side: [{100 - WINDOW}:100] vs [100:{100 + WINDOW}])")
    print("=" * 70)
    header = f"{'scenario':<13} {'var':<5} {'var_before':>12} {'var_after':>12} {'ratio':>8}"
    print(header)
    print("-" * len(header))
    for st, sc in scenarios.items():
        cp = sc.changepoint
        before = sc.data[cp - WINDOW:cp]
        after = sc.data[cp:cp + WINDOW]
        var_before = np.var(before, axis=0)
        var_after = np.var(after, axis=0)
        for i in range(N_NODES):
            ratio = var_after[i] / var_before[i] if var_before[i] > 0 else float("nan")
            print(f"{st:<13} X{i:<4} {var_before[i]:>12.4f} {var_after[i]:>12.4f} {ratio:>8.3f}")
        print("-" * len(header))


if __name__ == "__main__":
    main()
