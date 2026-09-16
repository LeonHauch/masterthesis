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

Robustness extension: sigma2 and gamma are NOT held fixed here. Per seed we
draw

    sigma2 ~ LogUniform(0.1, 5.0)
    gamma  = LogUniform(0.5, 3.0) * sqrt(sigma_W2)   (per tested edge --
             a median-heuristic-style scaling to that edge's own parent
             scale, via generate_matched_scenario's gamma_multiplier)

using two independent RNG streams (keyed off the seed but distinct from the
generator's own graph-structure RNG, so this randomization doesn't perturb
which nodes/edges get tested). sigma_W2 is drawn exactly as already
implemented in generator.py (from the random DAG). A single point in
(sigma2, gamma) parameter space would only prove the formulas work there --
this exercises the calibration across a wide range instead.

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

SIGMA2_RANGE = (0.1, 5.0)
GAMMA_MULT_RANGE = (0.5, 3.0)


def draw_seed_params(seed):
    """Per-seed sigma2 and gamma_multiplier, from RNG streams independent of
    the generator's own graph-structure RNG (which is seeded directly by `seed`)."""
    rng_sigma2 = np.random.default_rng(np.random.SeedSequence([seed, 0xA1]))
    rng_gamma = np.random.default_rng(np.random.SeedSequence([seed, 0xA2]))
    sigma2 = float(np.exp(rng_sigma2.uniform(np.log(SIGMA2_RANGE[0]), np.log(SIGMA2_RANGE[1]))))
    gamma_mult = float(np.exp(rng_gamma.uniform(np.log(GAMMA_MULT_RANGE[0]), np.log(GAMMA_MULT_RANGE[1]))))
    return sigma2, gamma_mult


def validate_below_threshold():
    print("=" * 96)
    print("BELOW THRESHOLD (Delta_a = 0.5 * delta_star) -- calibration must SUCCEED")
    print(f"sigma2 ~ LogUniform{SIGMA2_RANGE}, gamma = LogUniform{GAMMA_MULT_RANGE} * sqrt(sigma_W2), per seed")
    print("=" * 96)

    header = (
        f"{'seed':<5} {'child':<6} {'sigma2':>8} {'gamma':>8} {'sigma_W2':>10} {'delta_star':>11} "
        f"{'mech MMD2':>12} {'conf/noise MMD2':>16} {'abs diff':>10}"
    )
    print(header)
    print("-" * len(header))

    max_abs_diff = 0.0
    n_edges_checked = 0
    all_ok = True
    worst = None

    for seed in range(N_SEEDS):
        sigma2, gamma_mult = draw_seed_params(seed)
        kwargs = dict(
            n_nodes=N_NODES, T=T, seed=seed, strength_regime="below_threshold",
            base_noise_std=sigma2**0.5, gamma_multiplier=gamma_mult,
        )
        sc_mech = generate_matched_scenario(scenario_type="mechanism", **kwargs)
        sc_conf = generate_matched_scenario(scenario_type="confounding", **kwargs)
        sc_noise = generate_matched_scenario(scenario_type="noise", **kwargs)

        for e_mech, e_conf, e_noise in zip(sc_mech.tested_edges, sc_conf.tested_edges, sc_noise.tested_edges):
            assert e_mech.child == e_conf.child == e_noise.child
            assert abs(e_mech.delta_star - e_conf.delta_star) < 1e-9 * max(1.0, e_mech.delta_star)

            diff_conf = abs(e_mech.achieved_mmd2 - e_conf.achieved_mmd2)
            diff_noise = abs(e_mech.achieved_mmd2 - e_noise.achieved_mmd2)
            edge_max = max(diff_conf, diff_noise)
            if edge_max > max_abs_diff:
                max_abs_diff = edge_max
                worst = (seed, e_mech.child, e_mech.sigma2, e_mech.gamma, e_mech.sigma_W2)
            n_edges_checked += 1

            print(
                f"{seed:<5} {e_mech.child:<6} {e_mech.sigma2:>8.4f} {e_mech.gamma:>8.4f} "
                f"{e_mech.sigma_W2:>10.4f} {e_mech.delta_star:>11.4f} "
                f"{e_mech.achieved_mmd2:>12.6f} {e_conf.achieved_mmd2:>16.6f} {diff_conf:>10.2e}"
            )

    print("-" * len(header))
    print(f"Edges checked: {n_edges_checked}")
    print(f"Max |mechanism MMD2 - confound/noise MMD2| across all seeds/edges: {max_abs_diff:.3e}")
    if max_abs_diff < 1e-6:
        print("PASS: mechanism and confounding/noise MMD^2 match to numerical precision, across the")
        print("      full randomized (sigma2, gamma, sigma_W2) range -- not just one fixed point.")
    else:
        print(f"FAIL: mismatch exceeds tolerance. Worst case: seed={worst[0]}, child={worst[1]}, "
              f"sigma2={worst[2]:.6g}, gamma={worst[3]:.6g}, sigma_W2={worst[4]:.6g}")
        all_ok = False
    return all_ok


def validate_above_threshold():
    print()
    print("=" * 96)
    print("ABOVE THRESHOLD (Delta_a = 2.0 * delta_star) -- calibration must FAIL for confounding/noise")
    print(f"sigma2 ~ LogUniform{SIGMA2_RANGE}, gamma = LogUniform{GAMMA_MULT_RANGE} * sqrt(sigma_W2), per seed")
    print("=" * 96)

    header = (
        f"{'seed':<5} {'child':<6} {'sigma2':>8} {'gamma':>8} {'sigma_W2':>10} {'delta_star':>11} "
        f"{'mech MMD2':>12} {'sup=c(sigma2)':>14} {'gap':>10} {'conf raised?':>13} {'noise raised?':>14}"
    )
    print(header)
    print("-" * len(header))

    all_ok = True
    gaps = []
    failing = []

    for seed in range(N_SEEDS):
        sigma2, gamma_mult = draw_seed_params(seed)
        kwargs = dict(
            n_nodes=N_NODES, T=T, seed=seed, strength_regime="above_threshold",
            base_noise_std=sigma2**0.5, gamma_multiplier=gamma_mult,
        )
        sc_mech = generate_matched_scenario(scenario_type="mechanism", **kwargs)

        conf_raised = False
        try:
            generate_matched_scenario(scenario_type="confounding", **kwargs)
        except CalibrationImpossibleError:
            conf_raised = True

        noise_raised = False
        try:
            generate_matched_scenario(scenario_type="noise", **kwargs)
        except CalibrationImpossibleError:
            noise_raised = True

        if not (conf_raised and noise_raised):
            all_ok = False
            failing.append((seed, sigma2, gamma_mult, "calibration did not raise as expected"))

        for e in sc_mech.tested_edges:
            sup = c(e.sigma2, e.gamma)
            gap = e.achieved_mmd2 - sup
            gaps.append(gap)
            if gap <= 0:
                all_ok = False
                failing.append((seed, e.sigma2, e.gamma, f"child={e.child}: gap<=0 ({gap:.6g})"))
            print(
                f"{seed:<5} {e.child:<6} {e.sigma2:>8.4f} {e.gamma:>8.4f} {e.sigma_W2:>10.4f} "
                f"{e.delta_star:>11.4f} {e.achieved_mmd2:>12.6f} {sup:>14.6f} {gap:>10.6f} "
                f"{str(conf_raised):>13} {str(noise_raised):>14}"
            )

    print("-" * len(header))
    gaps = np.array(gaps)
    print(f"Gap (mechanism MMD2 - c(sigma2,gamma)) across all seeds/edges: "
          f"min={gaps.min():.4f}, max={gaps.max():.4f}, mean={gaps.mean():.4f}, std={gaps.std():.4f}")
    if all_ok:
        print("PASS: confounding/noise calibration failed as predicted (Satz B) on every seed, and every")
        print("      mechanism edge's MMD^2 exceeds the supremum c(sigma2, gamma) -- now over a real")
        print("      distribution of gaps rather than one constant value.")
    else:
        print("FAIL: unexpected result(s):")
        for f in failing:
            print(f"  {f}")
    return all_ok


def main():
    ok_below = validate_below_threshold()
    ok_above = validate_above_threshold()
    print()
    print("=" * 96)
    if ok_below and ok_above:
        print("VALIDATION PASSED for both regimes, under randomized sigma2/gamma/sigma_W2.")
    else:
        print("VALIDATION FAILED -- see above for the exact failing parameter combination.")
    print("=" * 96)


if __name__ == "__main__":
    main()
