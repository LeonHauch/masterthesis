"""
Phase 1 sanity check: generate one example of each matched scenario type
(mechanism / confounding / noise), below the MMD^2 threshold, and plot the
resulting (i.i.d.-across-time) series side by side, to visually confirm the
marginal shifts at the changepoint look comparably sized across the three --
they should only differ in their true cause (and in which nodes/edges are
tested, per generator.py's shared randomization), not in how big the
observable shift looks. See src/validate_mmd_calibration.py for the actual
quantitative (analytic MMD^2) calibration check.

Run: python src/plot_matched_scenarios.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generator import SCENARIO_TYPES, generate_matched_scenario


def main():
    n_nodes = 5
    T = 200
    seed = 0

    fig, axes = plt.subplots(len(SCENARIO_TYPES), 1, figsize=(10, 9), sharex=True)

    for ax, scenario_type in zip(axes, SCENARIO_TYPES):
        scenario = generate_matched_scenario(
            n_nodes=n_nodes, T=T, seed=seed, scenario_type=scenario_type, strength_regime="below_threshold"
        )
        for i in range(n_nodes):
            ax.plot(scenario.data[:, i], label=f"X{i}", alpha=0.8, linewidth=1)
        ax.axvline(scenario.changepoint, color="black", linestyle="--", linewidth=1)
        tested_children = [e.child for e in scenario.tested_edges]
        avg_mmd2 = sum(e.achieved_mmd2 for e in scenario.tested_edges) / len(scenario.tested_edges)
        ax.set_title(f"{scenario_type}  (tested children: {tested_children}, avg achieved MMD^2: {avg_mmd2:.4f})")
        ax.set_ylabel("value")

    axes[-1].set_xlabel("t")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Matched-counterfactual scenarios: same SEM, same-sized MMD^2 shift, different cause")
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "phase1_matched_scenarios.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
