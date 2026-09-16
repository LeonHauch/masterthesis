import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

np.random.seed(42)
T, N = 200, 3
data = np.random.randn(T, N)
for t in range(1, T):
    data[t, 1] += 0.5 * data[t - 1, 0]
    data[t, 2] += 0.5 * data[t - 1, 1]

dataframe = pp.DataFrame(data, var_names=[f"X{i}" for i in range(N)])
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
results = pcmci.run_pcmciplus(tau_max=2, pc_alpha=0.05)
print("PCMCI+ OK")
print("graph shape:", results["graph"].shape)
print(results["graph"][:, :, 0])

# --- J-PCMCI+ (context-aware) smoke test ---
from tigramite.jpcmciplus import JPCMCIplus

# 1 system var driven by 1 observed context var, plus a dummy time-context
N_sys = 2
data_j = np.random.randn(T, N_sys + 1)
for t in range(1, T):
    data_j[t, 1] += 0.6 * data_j[t - 1, 0] + 0.4 * data_j[t, 2]  # context influences X1

var_names_j = ["X0", "X1", "C0"]
dataframe_j = pp.DataFrame(
    data_j,
    var_names=var_names_j,
)

node_classification = {0: "system", 1: "system", 2: "space_context"}

jpcmci = JPCMCIplus(
    dataframe=dataframe_j,
    cond_ind_test=ParCorr(),
    node_classification=node_classification,
    verbosity=0,
)
results_j = jpcmci.run_jpcmciplus(tau_max=1, pc_alpha=0.05)
print("J-PCMCI+ OK")
print("graph shape:", results_j["graph"].shape)
