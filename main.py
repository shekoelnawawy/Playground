from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
import os
import warnings
from tqdm import tqdm
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# base_dir = '/Users/nawawy/Desktop/Research/Sepsis_data'
base_dir = '/home/mnawawy/Sepsis'
training_sets = ['training_setA', 'training_setB']
os.makedirs(os.path.join(base_dir, 'results', 'cluster_outputs'), exist_ok=True)

# get feature names
f = open(os.path.join(base_dir,'inputs/training_setA/p000001.psv'), 'r')
header = f.readline().strip()
features = np.array(header.split('|')[:-1])
f.close()

benign_data_df = pd.DataFrame()
adversarial_predictions_df = pd.DataFrame()

for training_set in training_sets:
    benign_data_path = os.path.join(base_dir, 'results', 'attack_outputs', training_set, 'Data', 'Benign')
    adversarial_predictions_path = os.path.join(base_dir,  'results', 'attack_outputs', training_set, 'Predictions', 'Adversarial')
    for f in tqdm(os.listdir(benign_data_path)):
        if os.path.isfile(os.path.join(benign_data_path, f)) and not f.lower().startswith('.') and f.lower().endswith('pkl'):
            benign_data = joblib.load(os.path.join(benign_data_path, f))
            per_patient_df = pd.DataFrame(benign_data)
            per_patient_df.columns = features
            per_patient_df.insert(loc=0, column='PatientID', value=f[:-4])
            benign_data_df = pd.concat([benign_data_df, per_patient_df], axis=0, ignore_index=True)
        else:
            raise Exception('Benign data file does not exist!')

        predictions_f = f[:-4] + '.psv'
        if os.path.isfile(os.path.join(adversarial_predictions_path, predictions_f)) and not predictions_f.lower().startswith('.') and predictions_f.lower().endswith('psv'):
            adversarial_predictions_file = open(os.path.join(adversarial_predictions_path, predictions_f), 'r')
            header = adversarial_predictions_file.readline().strip()
            per_patient_df = pd.DataFrame(np.loadtxt(adversarial_predictions_file, delimiter='|')[:, 1])
            adversarial_predictions_df = pd.concat([adversarial_predictions_df, per_patient_df], axis=0, ignore_index=True)
        else:
            raise Exception('Adversarial prediction file does not exist!')

# pre-processing
PatientIDs = benign_data_df['PatientID']
selected_features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender']
benign_data_df = benign_data_df.ffill()
benign_data_df = benign_data_df.fillna(0)
benign_data = np.array(benign_data_df.loc[:, selected_features])
benign_data = preprocessing.normalize(benign_data)
adversarial_output = np.array(adversarial_predictions_df)

# logistic regression for feature importance
# define dataset
X = benign_data
y = adversarial_output
# define the model
model = LogisticRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]

print(importance)



# import math
#
# import numpy as np
# import joblib
# from scipy import stats
# from dtaidistance import dtw, clustering
# import itertools
# from tqdm import tqdm
# import os
#
# base_dir = '/home/mnawawy/Downloads/OhioT1DM/processed_data/risk_profiling'
# # base_dir = '/Users/nawawy/Desktop/Research/OhioT1DM_data/risk_profiling'
# data_dir = os.path.join(base_dir, 'Data')
# out_dir = os.path.join(base_dir, 'cluster_outputs2')
# os.makedirs(out_dir, exist_ok=True)
#
# years = ['2018', '2020']
# patients_2018= ['559', '563', '570', '575', '588', '591']
# patients_2020 = ['540', '544', '552', '567', '584', '596']
# timeseries = []
#
# for year in years:
#     if year == '2018':
#         patients = patients_2018
#     else:
#         patients = patients_2020
#     for patient in patients:
#         instantaneous_error_path = os.path.join(data_dir, year, patient, 'instantaneous_error.pkl')
#         df = stats.zscore(np.array(joblib.load(instantaneous_error_path).mean(axis=1), dtype=np.double))
#         timeseries.append(df)
#
# numbers = []
# labels = []
# for i in range(len(patients_2018)):
#     numbers.append(i)
#     labels.append("A_" + str(i))
# for i in range(len(patients_2020)):
#     numbers.append(i+len(patients_2018))
#     labels.append("B_" + str(i))
#
#
# i=0
# dist = math.inf
# for x in itertools.permutations(numbers):
#     ts = []
#     lb = []
#     for j in range(len(x)):
#         ts.append(timeseries[int(x[j])])
#         lb.append(labels[int(x[j])])
#
#     ds = dtw.distance_matrix_fast(ts)
#     # print(ds)
#
#     # You can also pass keyword arguments identical to instantiate a Hierarchical object
#     model = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
#     cluster_idx = model.fit(ts)
#
#     # if model.linkage[-1][2] < dist:
#     dist = model.linkage[-1][2]
#     print('Minimum Distance so far:' + str(dist))
#     model.plot(os.path.join(out_dir, "clusters_"+str(i)+".pdf"), ts_label_margin = -400, show_ts_label=lb, show_tr_label=True)
#
#     i += 1
#
