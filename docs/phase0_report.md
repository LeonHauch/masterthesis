# Phase 0 — Environment Verification Report

Date: 2026-09-16

Goal: for each candidate method, verify it runs out of the box (or with
minimal fixes), and document minimal input/output format. No benchmark code
written yet, per Phase 0 scope.

## Summary table

| Method | Cloned | Installs cleanly | Runs on toy data | Notes |
|---|---|---|---|---|
| SPACETIME | yes | yes (`uv pip install -r src/requirements.txt`) | yes, but slow | see below |
| CASTOR (linear) | yes | yes (`uv pip install -r requirements.txt`) | yes, fast (~seconds) | see below |
| CASTOR (nonlinear) | yes | yes (same env) | yes, but slow | see below |
| FANTOM | **no** | n/a | n/a | repo not reachable — see "FANTOM" section |
| PCMCI+ / J-PCMCI+ (tigramite) | n/a (pip) | yes, with 1 fix | yes, fast (<1s for toy data) | see below |

## 1. SPACETIME

- Repo: `external/spacetime`, env: `external/spacetime/.venv` (uv, Python 3.11)
- Install: `uv pip install -r src/requirements.txt` — installed cleanly.
  Pulls in `torch` (with full CUDA wheels even though there's no GPU on this
  machine — ~3.9 GB venv), `tigramite`, `cdt` (Causal Discovery Toolbox),
  `causal-learn`, `causaldag`, `Rbeast`, `ruptures`. No R/rpy2 needed for the
  basic demo path.
- Entry point: `src/demo.py`. Runs a synthetic multi-regime time series
  generator (`exp.utils.gen_timeseries.gen_timeseries`) and feeds it to
  `st.spacetime.SpaceTime`.
- **Runs out of the box**, no code changes needed. However it is
  **very slow even on toy data**: the shipped demo (T=500, N=5 nodes, C=2
  contexts, R=3 regimes) took 25+ minutes of full single-core-pegged CPU time
  for the DAG-search phase alone on this machine (4 vCPUs). The change-point
  search (CPS) phase finishes quickly (seconds) and produces good regime
  recovery (F1 up to 1.0 in one run); it's the subsequent DAG search
  ("Phase 0/1/2" edge scoring) that dominates runtime.
  - Implication for the benchmark: toy scenarios should probably use fewer
    nodes/timesteps than the shipped demo, or budget significant wall-clock
    time per run.
- **Minimal input format**: a `Dataset`-like object (`data.datasets`) as
  produced by `gen_timeseries`; internally this is per-context per-regime
  numpy arrays of shape `(T, N)`. Using SPACETIME on our own generated data
  will require either (a) reusing `gen_timeseries`'s output structure, or
  (b) constructing the same `Dataset`/`sttypes` objects by hand — need to
  read `st/sttypes.py` and `st/dag_time.py` more closely in Phase 1.
- **Output format**: regime/change-point partition (list of `(start, len,
  regime_id)` tuples) + per-regime DAG(s), plus F1/precision/recall/ARI/NMI
  against ground truth when `truths` is supplied.

## 2. CASTOR (official, arahmani1/CASTOR)

- Repo: `external/CASTOR`, env: `external/CASTOR/.venv` (uv, Python 3.11)
- Install: `uv pip install -r requirements.txt` — installed cleanly (torch
  2.4 with CUDA wheels again pulled by default, ~5.4 GB venv despite no GPU).
- Entry point: `CASTOR_tutorial.ipynb` (also usable as plain `.py` via the
  `CASTOR` class import). Two code paths: `castor.run_linear(...)` and
  `castor.run_nonlinear(...)`.
- One fix needed to run standalone (not from inside the notebook): the
  tutorial calls `run_nonlinear(..., torch.device('cuda:0'), ...)`, which
  fails without a GPU. `device` is a plain parameter of `CASTOR.run_nonlinear`
  (`CASTOR.py:95`), so passing `torch.device('cpu')` works with **no other
  code changes**.
- **Linear CASTOR runs out of the box and is fast**: a smoke test with 2
  regimes, 3 nodes, 80+80 samples completed in a few seconds
  (`castor.run_linear(2, 1, 0.4, 20, 20)` → returns `model_n, graphss,
  gamma_hat, L`).
- **Nonlinear CASTOR runs but is slow**: same toy size (2 regimes, 3 nodes,
  160 samples total) took well over several minutes of CPU time (trains an
  MLP per regime candidate via L-BFGS). Final confirmation pending — see
  "Still running" below.
- **Minimal input format**: `CASTOR(data: pd.DataFrame, X_syn: np.ndarray,
  X_lag_syn: np.ndarray, lags=2)`, where `X_syn` is the "current time step"
  block (`N` columns) and `X_lag_syn` is the lagged block (`N * lags`
  columns, laid out as `<node>_lag1`, `<node>_lag2`, …). The repo's own
  `data_generation.simulate_regime_MTS(...)` produces a DataFrame with
  `<node>_lag0`, `<node>_lag1`, … columns already in this shape — we can
  reuse the same column convention for our injected data.
- **Output format**: `run_linear` → `(model_n, graphss, gamma_hat, L)` where
  `gamma_hat` is a `(T, n_regimes)` soft regime-assignment matrix and `L` is
  a `(lag*N, lag*N, n_candidate_graphs)` tensor of (thresholdable) edge
  weights per candidate graph. `run_nonlinear` → `(gamma_hat, model_n)` where
  `model_n` is a list of trained per-regime MLP models exposing weight
  tensors (e.g. `.W_no_thres`, `.A_no_thres`) that get thresholded into
  adjacency matrices (see tutorial's plotting cell).

## 3. FANTOM (arahmani1/fantom)

**Not verified — repo is not reachable from this session.**

- `git clone https://github.com/arahmani1/fantom.git` fails with
  `fatal: could not read Username for 'https://github.com': terminal
  prompts disabled` — the same error occurs for several name variants
  (`FANTOM`, `Fantom`, `fantom-causal`, `fantom_release`), which is the
  generic response git/GitHub give for both "private, no access" and
  "doesn't exist" (to avoid leaking which one).
- Direct unauthenticated fetch of `github.com/arahmani1/fantom` returns
  HTTP 404.
- The `arahmani1` GitHub profile's visible repository list (fetched via
  unauthenticated page load) shows only 4 repos: `arahmani1.github.io`,
  `RLAD`, `CASTOR`, `graphsexercises` — **no `fantom`**.
- A NeurIPS 2025 paper by the same authors (Rahmani & Frossard, "Flow based
  approach for Dynamic Temporal Causal models with non-Gaussian or
  Heteroscedastic Noises" — this appears to be the FANTOM paper) cites
  `https://github.com/arahmani1/fantom.git` as the code repository, so the
  URL you gave is very likely correct — the repo is just not (yet) public.
- I tried to request access via this session's `add_repo` tool, but it
  refused: this session is scoped to the `leonhauch/masterthesis` GitHub
  org/owner, and `add_repo` doesn't support attaching a repo from a
  different owner (`arahmani1`) to an already-scoped session.

**What would unblock this**: either (a) the authors make the repo public,
(b) you have another access path (e.g. you were sent a private link, or
know a mirror/fork), or (c) we start a **separate** Claude Code session
with `arahmani1/fantom` as its *initial* repo source — that might succeed if
the repo exists but is merely permission-gated for this GitHub App
install, though if it's genuinely private this likely won't work either
without you granting access.

## 4. PCMCI+ / J-PCMCI+ (tigramite baseline)

- Env: `external/tigramite_env` (uv, Python 3.11), installed via
  `pip install tigramite` (not cloned — this is the intended pip package
  usage per your request).
- **One fix needed**: `pip install tigramite` alone is missing a runtime
  dependency — `tigramite.pcmci_base` imports `joblib`, which is not pulled
  in automatically. Fixed with `uv pip install joblib networkx matplotlib
  scikit-learn numba` (some of these may not all be strictly required for
  PCMCI+/J-PCMCI+ specifically, but this covers the common import paths).
- **Runs out of the box after that fix, and is fast**: both
  `PCMCI.run_pcmciplus(...)` and `JPCMCIplus.run_jpcmciplus(...)` completed
  in well under a second on toy data (T=200, 3 variables).
- **Minimal input format**: `tigramite.data_processing.DataFrame(data:
  np.ndarray of shape (T, N), var_names=[...])`. For J-PCMCI+, additionally
  a `node_classification` dict mapping each variable index to one of
  `"system"`, `"time_context"`, `"space_context"` (observed context
  variables vs. system variables you want the causal graph over).
- **Output format**: `results["graph"]`, an `(N, N, tau_max+1)` array of
  edge-mark strings (e.g. `"-->"`, `"<--"`, `""` for no edge) — the standard
  tigramite graph representation, plus `results["val_matrix"]` /
  `results["p_matrix"]` for edge strengths/significance.
- As expected for a **stationary baseline**, plain PCMCI+ has no concept of
  regimes at all — it will just fit one graph to the whole series. J-PCMCI+
  can take an observed regime/context indicator as a `space_context` or
  `time_context` variable, but that requires *knowing* the regime labels or
  a context variable ahead of time, unlike SPACETIME/CASTOR which infer
  regimes unsupervised. This is exactly the "negative control" role we want
  it to play in the benchmark.

## Disk footprint (uv venvs)

- `external/CASTOR/.venv`: 5.4 GB (torch w/ CUDA wheels despite no GPU)
- `external/spacetime/.venv`: 3.9 GB (same)
- `external/tigramite_env`: 437 MB

For future sessions, consider installing CPU-only torch wheels (`--index-url
https://download.pytorch.org/whl/cpu`) to cut disk usage substantially if
GPU is never available in the dev environment.

## Open items / still running at time of writing

- SPACETIME's `demo.py` and CASTOR's nonlinear smoke test were both still
  running in the background when this report was drafted (both are
  CPU-bound and slow even on tiny toy inputs). Will confirm final
  success/failure and append results once they complete.
- FANTOM needs your input on how to proceed (see above).
