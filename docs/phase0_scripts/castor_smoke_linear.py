import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from data_generation import simulate_regime_MTS
from utils import identity
from CASTOR import CASTOR

g_list, df_total, intra_nodes, inter_nodes = simulate_regime_MTS(
    regime=2,
    func_l=[identity, identity],
    num_nodes=3,
    n_samples=[80, 80],
    p=1,
    degree_intra=2,
    degree_inter=1,
    graph_type_intra="barabasi-albert",
    graph_type_inter="erdos-renyi",
    sem_type="linear-gauss",
    noise_scale=1,
    ANM_type="Linear",
    max_data_gen_trials=1000,
)

n_nodes = 3
rearange_intra = [str(i) + "_lag0" for i in range(n_nodes)]
rearange_inter = [str(i) + "_lag1" for i in range(n_nodes)]

X_syn = df_total[rearange_intra].to_numpy()
X_lag_syn = df_total[rearange_inter].to_numpy()
data = pd.DataFrame(X_syn)

castor = CASTOR(data, X_syn, X_lag_syn, lags=2)
model_n, graphss, gamma_hat, L = castor.run_linear(2, 1, 0.4, 20, 20)

print("OK: run_linear completed")
print("gamma_hat shape:", np.array(gamma_hat).shape)
print("L shape:", np.array(L).shape)
print("num graphs:", len(graphss))
