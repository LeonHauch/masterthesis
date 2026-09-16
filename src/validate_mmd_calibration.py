"""
Phase 1 validation (Step 5) for the MMD^2-calibrated matched-counterfactual
generator (src/generator.py). Prints, across 20 seeds, for both threshold
regimes:

- below_threshold (Delta_a = 0.5 * delta_star): calibration must succeed,
  and the mechanism scenario's analytic MMD^2 must match the solved
  confounding/noise delta's analytic MMD^2 almost exactly.
- above_threshold (Delta_a = 2.0 * delta_star): calibration must FAIL for
  confounding/noise (CalibrationImpossibleError), and the mechanism
  scenario's MMD^2 must exceed the proven supremum c(sigma2, gamma).

All MMD^2 values are computed analytically from the closed-form formulas
(not estimated from simulated data) -- this checks the calibration
machinery itself, not sampling noise.

Run: python src/validate_mmd_calibration.py
"""
import numpy as np

from generator import (
    CalibrationImpossibleError,
    SCENARIO_TYPES,
    c,
    generate_matched_scenario,
)

N_NODES = 5
T = 50
N_SEEDS = 20


def validate_below_threshold():
    print("=" * 78)
    print("BELOW THRESHOLD (Delta_a = 0.5 * delta_star) -- calibration must SUCCEED")
    print("=" * 78)

    header = (
        f"{'seed':<5} {'child':<6} {'delta_star':>11} {'Delta_a=0.5*ds':>15} "
        f"{'mech MMD2':>12} {'solved delta':>13} {'conf/noise MMD2':>16} {'abs diff':>10}"
    )
    print(header)
    print("-" * len(header))

    max_abs_diff = 0.0
    n_edges_checked = 0
    all_ok = True

    for seed in range(N_SEEDS):
        sc_mech = generate_matched_scenario(
            n_nodes=N_NODES, T=T, seed=seed, scenario_type="mechanism", strength_regime="below_threshold"
        )
        sc_conf = generate_matched_scenario(
            n_nodes=N_NODES, T=T, seed=seed, scenario_type="confounding", strength_regime="below_threshold"
        )
        sc_noise = generate_matched_scenario(
            n_nodes=N_NODES, T=T, seed=seed, scenario_type="noise", strength_regime="below_threshold"
        )

        for e_mech, e_conf, e_noise in zip(sc_mech.tested_edges, sc_conf.tested_edges, sc_noise.tested_edges):
            assert e_mech.child == e_conf.child == e_noise.child
            assert abs(e_mech.delta_star - e_conf.delta_star) < 1e-9  # shared randomization check

            diff_conf = abs(e_mech.achieved_mmd2 - e_conf.achieved_mmd2)
            diff_noise = abs(e_mech.achieved_mmd2 - e_noise.achieved_mmd2)
            max_abs_diff = max(max_abs_diff, diff_conf, diff_noise)
            n_edges_checked += 1

            print(
                f"{seed:<5} {e_mech.child:<6} {e_mech.delta_star:>11.4f} {e_mech.delta_a_target:>15.4f} "
                f"{e_mech.achieved_mmd2:>12.6f} {e_conf.delta_confound_or_noise:>13.4f} "
                f"{e_conf.achieved_mmd2:>16.6f} {diff_conf:>10.2e}"
            )

    print("-" * len(header))
    print(f"Edges checked: {n_edges_checked}")
    print(f"Max |mechanism MMD2 - confound/noise MMD2| across all seeds/edges: {max_abs_diff:.3e}")
    if max_abs_diff < 1e-6:
        print("PASS: mechanism and confounding/noise MMD^2 match to numerical precision.")
    else:
        print("FAIL: mismatch exceeds tolerance.")
        all_ok = False
    return all_ok


def validate_above_threshold():
    print()
    print("=" * 78)
    print("ABOVE THRESHOLD (Delta_a = 2.0 * delta_star) -- calibration must FAIL for confounding/noise")
    print("=" * 78)

    header = (
        f"{'seed':<5} {'child':<6} {'delta_star':>11} {'Delta_a=2*ds':>13} "
        f"{'mech MMD2':>12} {'sup=c(sigma2)':>14} {'gap':>10} {'conf raised?':>13} {'noise raised?':>14}"
    )
    print(header)
    print("-" * len(header))

    all_ok = True
    for seed in range(N_SEEDS):
        sc_mech = generate_matched_scenario(
            n_nodes=N_NODES, T=T, seed=seed, scenario_type="mechanism", strength_regime="above_threshold"
        )

        conf_raised = False
        try:
            generate_matched_scenario(
                n_nodes=N_NODES, T=T, seed=seed, scenario_type="confounding", strength_regime="above_threshold"
            )
        except CalibrationImpossibleError:
            conf_raised = True

        noise_raised = False
        try:
            generate_matched_scenario(
                n_nodes=N_NODES, T=T, seed=seed, scenario_type="noise", strength_regime="above_threshold"
            )
        except CalibrationImpossibleError:
            noise_raised = True

        if not (conf_raised and noise_raised):
            all_ok = False

        for e in sc_mech.tested_edges:
            sup = c(e.sigma2, e.gamma)
            gap = e.achieved_mmd2 - sup
            if gap <= 0:
                all_ok = False
            print(
                f"{seed:<5} {e.child:<6} {e.delta_star:>11.4f} {e.delta_a_target:>13.4f} "
                f"{e.achieved_mmd2:>12.6f} {sup:>14.6f} {gap:>10.6f} "
                f"{str(conf_raised):>13} {str(noise_raised):>14}"
            )

    print("-" * len(header))
    if all_ok:
        print("PASS: confounding/noise calibration failed as predicted (Satz B) on every seed, and every")
        print("      mechanism edge's MMD^2 exceeds the supremum c(sigma2, gamma) as expected.")
    else:
        print("FAIL: either a calibration unexpectedly succeeded, or a mechanism MMD^2 did not exceed the supremum.")
    return all_ok


def main():
    ok_below = validate_below_threshold()
    ok_above = validate_above_threshold()
    print()
    print("=" * 78)
    if ok_below and ok_above:
        print("VALIDATION PASSED for both regimes.")
    else:
        print("VALIDATION FAILED -- see above.")
    print("=" * 78)


if __name__ == "__main__":
    main()
