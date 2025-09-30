import math

import numpy as np
import joblib
from scipy import stats
from dtaidistance import dtw, clustering
import itertools
from tqdm import tqdm
import os

base_dir = '/home/mnawawy/Downloads/OhioT1DM/processed_data/risk_profiling'
# base_dir = '/Users/nawawy/Desktop/Research/OhioT1DM_data/risk_profiling'
data_dir = os.path.join(base_dir, 'Data')
out_dir = os.path.join(base_dir, 'cluster_outputs')
os.makedirs(out_dir, exist_ok=True)

years = ['2018', '2020']
patients_2018= ['559', '563', '570', '575', '588', '591']
patients_2020 = ['540', '544', '552', '567', '584', '596']
timeseries = []

for year in years:
    if year == '2018':
        patients = patients_2018
    else:
        patients = patients_2020
    for patient in patients:
        instantaneous_error_path = os.path.join(data_dir, year, patient, 'instantaneous_error.pkl')
        df = stats.zscore(np.array(joblib.load(instantaneous_error_path).mean(axis=1), dtype=np.double))
        timeseries.append(df)

numbers = []
labels = []
for i in range(len(patients_2018)+len(patients_2020)):
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

    ds = dtw.distance_matrix_fast(ts)
    # print(ds)

    # You can also pass keyword arguments identical to instantiate a Hierarchical object
    model = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
    cluster_idx = model.fit(ts)

    if model.linkage[-1][2] < dist:
        dist = model.linkage[-1][2]
        print('Minimum Distance so far:' + str(dist))
        model.plot(os.path.join(out_dir, "clusters.pdf"), ts_label_margin = -200, show_ts_label=lb, show_tr_label=True)

    i += 1

