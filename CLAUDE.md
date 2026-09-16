# Master's Thesis: Regime-Aware Causal Discovery Benchmark

Last updated: 2026-09-16 (Phase 0)

## Goal

Build a benchmark that tests whether regime-aware causal discovery methods
can correctly distinguish three different underlying causes of a regime
change in multivariate time series, when the *observable* distributional
shift looks similar across all three ("matched counterfactual" design):

1. **TRUE** — a genuine causal mechanism change (an edge appears/disappears,
   or an edge's functional form/parameters change).
2. **CONFOUNDING** — the shift is driven by a change in an unobserved
   confounder, with no actual change to the causal graph among observed
   variables.
3. **NOISE** — the shift is just a change in exogenous noise
   distribution/variance, again with no true graph change.

The benchmark generates matched synthetic scenarios (same marginal/observed
distributional shift signature) under each of the three regimes, and checks
whether each method reports a difference in causal structure only for the
TRUE condition — a method that can't tell TRUE from CONFOUNDING/NOISE is
prone to false positives about "mechanism change" in real applications
(e.g. flagging a policy effect when it was really a confounder shift).

## Methods under test

| Method | Repo | Role |
|---|---|---|
| SPACETIME | https://github.com/srhmm/spacetime | Regime-aware causal discovery (change-point search + DAG search) |
| CASTOR | https://github.com/arahmani1/CASTOR | Regime-aware causal discovery via EM (linear + nonlinear variants) |
| FANTOM | https://github.com/arahmani1/fantom | Regime-aware causal discovery w/ normalizing flows + Bayesian EM (NOT YET VERIFIED — repo inaccessible, see Phase 0 report) |
| PCMCI+ / J-PCMCI+ | `pip install tigramite` | Stationary (PCMCI+) and context-aware (J-PCMCI+) baseline — expected to NOT distinguish the three conditions well, serving as a negative control |

## Repo layout

```
masterthesis/
  src/            benchmark code (data generators, adapters, metrics) — empty in Phase 0
  data/           generated datasets — empty in Phase 0
  results/        benchmark outputs — empty in Phase 0
  docs/           write-ups, phase reports
  external/       cloned third-party method repos + per-method venvs (gitignored)
    spacetime/        (cloned, has .venv)
    CASTOR/            (cloned, has .venv)
    tigramite_env/     (venv only, tigramite installed via pip)
    fantom/            NOT cloned — access blocked, see docs/phase0_report.md
```

Each external method gets its **own `uv` venv** (`external/<repo>/.venv`) because
their dependency pins conflict (different numpy/torch/pandas versions). Never
install method dependencies into a shared/root environment.

## Status

- Phase 0 (environment verification) — see `docs/phase0_report.md` for full
  findings per method (install steps, minimal I/O format, what broke).
- Phase 1+ (actual benchmark design/implementation) — not started.

## Conventions for future sessions

- Do not commit `external/*/.venv` or `external/*/data` artifacts — add to
  `.gitignore`.
- FANTOM access needs to be resolved before it can be benchmarked (either the
  authors make the repo public, or the user provides another way to access
  it — a fresh session scoped to `arahmani1/fantom` as the initial repo may
  be able to reach it if it does exist and is just permission-gated).
