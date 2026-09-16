# Reproducing `external/` (gitignored)

`external/` holds third-party method repos and their per-method virtual
environments. It's gitignored (nested `.git` dirs would otherwise turn into
awkward submodule links, and the venvs are multi-GB). To reproduce:

```bash
mkdir -p external && cd external

# SPACETIME
git clone https://github.com/srhmm/spacetime.git
cd spacetime && uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r src/requirements.txt
deactivate && cd ..

# CASTOR
git clone https://github.com/arahmani1/CASTOR.git
cd CASTOR && uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
deactivate && cd ..

# tigramite baseline (PCMCI+ / J-PCMCI+) — pip package, no clone
uv venv tigramite_env --python 3.11 && source tigramite_env/bin/activate
uv pip install tigramite joblib networkx matplotlib scikit-learn numba
deactivate

# FANTOM — NOT YET AVAILABLE, see docs/phase0_report.md
```

The verification smoke-test scripts used in Phase 0 are kept under
`docs/phase0_scripts/` (copy them into the relevant `external/<repo>/`
directory before running, since they import from that repo's modules).

Note: both `spacetime`'s and CASTOR's `requirements.txt`/tutorial pull in
CUDA-enabled `torch` wheels by default (multi-GB) even with no GPU present.
Use `--index-url https://download.pytorch.org/whl/cpu` for a much smaller
CPU-only install if disk space is a concern.
