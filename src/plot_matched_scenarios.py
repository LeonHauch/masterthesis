"""
Phase 1 sanity check: generate one example of each matched scenario type
(mechanism / confounding / noise) and plot the resulting time series side by
side, to visually confirm the marginal shifts at the changepoint look
comparably sized across the three -- they should only differ in their true
cause, not in how big the observable shift looks.

Run: python src/plot_matched_scenarios.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generator import SCENARIO_TYPES, generate_matched_scenario


def main():
    n_nodes = 4
    T = 200
    seed = 0

    fig, axes = plt.subplots(len(SCENARIO_TYPES), 1, figsize=(10, 9), sharex=True)

    for ax, scenario_type in zip(axes, SCENARIO_TYPES):
        scenario = generate_matched_scenario(n_nodes=n_nodes, T=T, seed=seed, scenario_type=scenario_type)
        for i in range(n_nodes):
            ax.plot(scenario.data[:, i], label=f"X{i}", alpha=0.8, linewidth=1)
        ax.axvline(scenario.changepoint, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{scenario_type}  (achieved avg-variance shift: {scenario.shift_size:.3f})")
        ax.set_ylabel("value")

    axes[-1].set_xlabel("t")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Matched-counterfactual scenarios: same DAG, same-sized shift, different cause")
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "phase1_matched_scenarios.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
