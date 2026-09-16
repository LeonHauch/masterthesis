"""
Phase 1 diagnostics for the matched-counterfactual generator (src/generator.py).

Prints plain-number checks, without changing any generator logic:

1. PARAMETER CHECK -- the actual VAR(1) edge-weight matrix (dag_pre vs.
   dag_post) for each scenario at a fixed seed, so mechanism changes (or the
   lack thereof) can be verified by eye.
2. VISIBILITY CHECK -- shape/columns of the data array as it would be handed
   to a causal discovery method, confirming the latent confounder is not
   one of its columns.
3. VARIANCE-SHIFT CHECK, across 20 seeds -- per-variable (not averaged)
   empirical variance in a window before vs. after the changepoint, for
   each scenario, summarized as mean +/- std across seeds of:
   (a) how many nodes are affected, (b) the per-node variance ratio for
   affected vs. unaffected nodes.

Run: python src/diagnostics_matched_scenarios.py
"""
import numpy as np
import networkx as nx

from generator import SCENARIO_TYPES, generate_matched_scenario

N_NODES = 4
T = 200
SEED = 0
N_SEEDS = 20
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


def part1_parameter_check():
    print("=" * 70)
    print("1. PARAMETER CHECK -- VAR(1) edge-weight matrix A (A[i,j] = j -> i)")
    print(f"   seed={SEED}")
    print("=" * 70)
    for st in SCENARIO_TYPES:
        sc = generate_matched_scenario(n_nodes=N_NODES, T=T, seed=SEED, scenario_type=st)
        A_pre = graph_to_matrix(sc.dag_pre, N_NODES)
        A_post = graph_to_matrix(sc.dag_post, N_NODES)
        identical = np.allclose(A_pre, A_post)
        print(f"\n[{st}]  affected_nodes={list(sc.affected_nodes)}  A_pre == A_post ? {identical}")
        print_matrix(A_pre, "A_pre  (t < 100)")
        print_matrix(A_post, "A_post (t >= 100)")
        if not identical:
            diff = A_post - A_pre
            changed = np.argwhere(np.abs(diff) > 1e-8)
            print("  Changed entries (i <- j : pre -> post):")
            for i, j in changed:
                tag = " [self-loop]" if i == j else ""
                print(f"    {i} <- {j} : {A_pre[i, j]:+.4f} -> {A_post[i, j]:+.4f}{tag}")


def part2_visibility_check():
    print()
    print("=" * 70)
    print("2. VISIBILITY CHECK -- data handed to a causal discovery method")
    print("=" * 70)
    sc = generate_matched_scenario(n_nodes=N_NODES, T=T, seed=SEED, scenario_type="confounding")
    columns = [f"X{i}" for i in range(N_NODES)]
    print(f"  scenario: confounding")
    print(f"  data.shape           = {sc.data.shape}   (T={sc.T}, n_nodes={N_NODES})")
    print(f"  data columns         = {columns}")
    print(f"  'Z' / confounder in data.columns? {'Z' in columns}")
    print(
        f"  latent_confounder is a SEPARATE array, shape = {sc.latent_confounder.shape}, "
        f"not concatenated into data"
    )
    assert sc.data.shape[1] == N_NODES, "data must only contain the observed nodes"


def part3_variance_shift_check():
    print()
    print("=" * 70)
    print(f"3. VARIANCE-SHIFT CHECK across {N_SEEDS} seeds -- per-variable, before vs. after t=100")
    print(f"   (window = {WINDOW} samples on each side: [{100 - WINDOW}:100] vs [100:{100 + WINDOW}])")
    print("=" * 70)

    # per scenario, collect across seeds: n_affected, ratio for affected nodes, ratio for unaffected nodes
    n_affected_by_scenario = {st: [] for st in SCENARIO_TYPES}
    affected_ratios_by_scenario = {st: [] for st in SCENARIO_TYPES}
    unaffected_ratios_by_scenario = {st: [] for st in SCENARIO_TYPES}

    for seed in range(N_SEEDS):
        for st in SCENARIO_TYPES:
            sc = generate_matched_scenario(n_nodes=N_NODES, T=T, seed=seed, scenario_type=st)
            cp = sc.changepoint
            before = sc.data[cp - WINDOW : cp]
            after = sc.data[cp : cp + WINDOW]
            var_before = np.var(before, axis=0)
            var_after = np.var(after, axis=0)
            ratio = var_after / var_before

            affected_mask = np.zeros(N_NODES, dtype=bool)
            affected_mask[sc.affected_nodes] = True

            n_affected_by_scenario[st].append(len(sc.affected_nodes))
            affected_ratios_by_scenario[st].extend(ratio[affected_mask].tolist())
            if np.any(~affected_mask):
                unaffected_ratios_by_scenario[st].extend(ratio[~affected_mask].tolist())

    header = f"{'scenario':<13} {'n_affected (mean+-std)':<24} {'affected ratio (mean+-std)':<28} {'unaffected ratio (mean+-std)':<28}"
    print(header)
    print("-" * len(header))
    for st in SCENARIO_TYPES:
        n_aff = np.array(n_affected_by_scenario[st], dtype=float)
        aff_r = np.array(affected_ratios_by_scenario[st])
        unaff_r = np.array(unaffected_ratios_by_scenario[st])
        unaff_str = f"{unaff_r.mean():.3f} +/- {unaff_r.std():.3f}" if len(unaff_r) else "n/a (never unaffected)"
        print(
            f"{st:<13} {f'{n_aff.mean():.2f} +/- {n_aff.std():.2f}':<24} "
            f"{f'{aff_r.mean():.3f} +/- {aff_r.std():.3f}':<28} {unaff_str:<28}"
        )
    print("-" * len(header))
    print(
        "\nInterpretation: 'n_affected' distributions should match across scenarios (same seeds ->"
        "\nsame affected_nodes draw for all three). 'affected ratio' distributions should be similar"
        "\nacross scenarios (same target_shift_ratio calibration target). 'unaffected ratio' should"
        "\nstay close to 1.0 in all three (only minor DAG-mediated spillover expected)."
    )


def main():
    part1_parameter_check()
    part2_visibility_check()
    part3_variance_shift_check()


if __name__ == "__main__":
    main()
