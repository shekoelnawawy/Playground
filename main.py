import math

import numpy as np
import joblib
from scipy import stats
from dtaidistance import dtw, clustering
import itertools
from tqdm import tqdm
import os

base_dir = '/home/mnawawy/Downloads/Sepsis/processed_data/risk_profiling'
# base_dir = '/Users/nawawy/Desktop/Research/Sepsis_data/risk_profiling'
data_dir = os.path.join(base_dir, 'Data')
out_dir = os.path.join(base_dir, 'cluster_outputs')
os.makedirs(out_dir, exist_ok=True)

timeseries = []


risk_profiles_path = os.path.join(data_dir, 'RiskProfiles.pkl')
risk_profiles = joblib.load(risk_profiles_path)

for i in range(len(risk_profiles)):
    df = stats.zscore(np.array(risk_profiles[i][1]))
    timeseries.append(df)


numbers = []
labels = []
for i in range(len(risk_profiles)):
    numbers.append(i)
    labels.append("p"+str(i))

i=0
dist = math.inf

for x in itertools.permutations(numbers):
    ts = []
    lb = []
    for j in range(len(x)):
        ts.append(timeseries[int(x[j])])
        lb.append(labels[int(x[j])])

    ds = dtw.distance_matrix_fast(ts, compact=True)
    print(ds)
    ds[np.isnan(ds)] = -1
    print(np.sort(ds))

    # You can also pass keyword arguments identical to instantiate a Hierarchical object
    model = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
    cluster_idx = model.fit(ts)

    if model.linkage[-1][2] < dist:
        dist = model.linkage[-1][2]
        print('Minimum Distance so far:' + str(dist))
        model.plot(os.path.join(out_dir, "clusters.pdf"), ts_label_margin = -200, show_ts_label=lb, show_tr_label=True)

    i += 1

