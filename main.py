from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# base_dir = '/Users/nawawy/Desktop/Research/Sepsis_data'
base_dir = '/home/mnawawy/Sepsis'
training_sets = ['training_setA', 'training_setB']
os.makedirs(os.path.join(base_dir, 'results', 'cluster_outputs'), exist_ok=True)

# get feature names
f = open(os.path.join(base_dir,'physionet.org/files/challenge-2019/1.0.0/training/training_setA/p000001.psv'), 'r')
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

timeseries = importance * benign_data

risk_profiles = []
risk_scores = []
old_PatientID = PatientIDs[0]
for j in range(len(timeseries)):
    if PatientIDs[j] == old_PatientID:
        risk_scores.append(float(sum(timeseries[j,:])))
        if j == len(timeseries)-1:
            risk_profiles.append((old_PatientID, risk_scores))
    else:
        risk_profiles.append((old_PatientID, risk_scores))
        old_PatientID = PatientIDs[j]
        risk_scores = []
        risk_scores.append(float(sum(timeseries[j,:])))


joblib.dump(risk_profiles, os.path.join(base_dir, 'results', 'cluster_outputs', 'RiskProfiles.pkl'))

unique_PatientIDs = [item[0] for item in risk_profiles]
df = pd.DataFrame([item[1] for item in risk_profiles]).ffill(axis=1)
model = KMeans(n_clusters=2, verbose = 1)
model.fit(df)
cluster_centers = model.cluster_centers_
print(cluster_centers)

from sklearn.metrics.pairwise import euclidean_distances
distances_matrix = euclidean_distances(cluster_centers)
print("Distance matrix between cluster centers:\n", distances_matrix)


# import os
# from sklearn.svm import OneClassSVM
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# from sklearn.model_selection import KFold, train_test_split
# import numpy as np
# import joblib
# import random
# import warnings
# import math
# import pandas as pd
# from tqdm import tqdm
#
# warnings.filterwarnings('ignore')
#
# base_dir = '/home/mnawawy/Sepsis'
# # base_dir = '/Users/nawawy/Desktop/Research/Sepsis_data'
# output_path = os.path.join(base_dir, 'Defenses', 'ResultsOneClassSVM')
#
#
# AllPatientsData = joblib.load(os.path.join(base_dir, 'results', 'attack_outputs', 'adversarial_data.pkl'))
#
# clf = OneClassSVM(kernel='poly',degree=5,gamma='auto',cache_size=4096, verbose=True, max_iter = 5000)
#
#
# AllPatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'AllPatientIDs.pkl'))
# MostVulnerablePatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'threshold_10', 'MostVulnerablePatientIDs.pkl'))
# # MoreVulnerablePatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'threshold_5', 'MoreVulnerablePatientIDs.pkl'))
# # LessVulnerablePatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'threshold_5', 'LessVulnerablePatientIDs.pkl'))
# MoreVulnerablePatientIDs = MostVulnerablePatientIDs
# LessVulnerablePatientIDs = list(set(AllPatientIDs) - set(MostVulnerablePatientIDs))
# ######################################################################################################################################
# # # Samples
# # os.makedirs(os.path.join(output_path, 'SamplesTraining'), exist_ok=True)
# # results = open(os.path.join(output_path, 'SamplesTraining', 'Results.csv'), 'w')
# # results.write('Run,Accuracy,Precision,Recall,F1\n')
# #
# # Accuracy = []
# # Precision = []
# # Recall = []
# # F1 = []
# # for run in range(5):
# #     print('SamplesTraining\tTrial: ' + str(run))
# #
# #     split = train_test_split(AllPatientIDs, train_size=len(LessVulnerablePatientIDs))
# #     train_indices = split[0]
# #     test_indices = split[1][:math.floor(0.2 * len(AllPatientIDs))]
# #
# #     train = AllPatientsData[AllPatientsData['PatientID'].isin(train_indices)].drop(columns=['PatientID']).to_numpy()
# #     train_x = train[:, :-1]
# #     train_y = train[:, -1].astype(int)
# #
# #     test = AllPatientsData[AllPatientsData['PatientID'].isin(test_indices)].drop(columns=['PatientID']).to_numpy()
# #     test_x = test[:, :-1]
# #     test_y = test[:, -1].astype(int)
# #
# #     clf.fit(train_x, train_y)
# #
# #     lst = clf.predict(test_x)
# #     lst[lst == 1] = 0
# #     lst[lst == -1] = 1
# #
# #     Accuracy.insert(len(Accuracy), accuracy_score(test_y, lst) * 100)
# #     Precision.insert(len(Precision), precision_score(test_y, lst))
# #     Recall.insert(len(Recall), recall_score(test_y, lst))
# #     F1.insert(len(F1), f1_score(test_y, lst))
# #
# #     results.write(
# #         str(run) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
# #             recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
# #
# # results.write(
# #     'Average,' + str(np.mean(Accuracy)) + ',' + str(np.mean(Precision)) + ',' + str(np.mean(Recall)) + ',' + str(
# #         np.mean(F1)) + '\n')
# # results.close()
# # ######################################################################################################################################
# # # All patients
# # os.makedirs(os.path.join(output_path, 'All'), exist_ok=True)
# # results = open(os.path.join(output_path, 'All', 'Results.csv'), 'w')
# # results.write('Cross-Validation,Accuracy,Precision,Recall,F1\n')
# #
# # cv = 0
# # kf = KFold(n_splits=5)
# # Accuracy = []
# # Precision = []
# # Recall = []
# # F1 = []
# # for train_indices, test_indices in kf.split(AllPatientIDs):
# #     print('All\tCV: ' + str(cv))
# #
# #     train = AllPatientsData[AllPatientsData['PatientID'].isin([AllPatientIDs[i] for i in train_indices])].drop(columns=['PatientID']).to_numpy()
# #     train_x = train[:, :-1]
# #     train_y = train[:, -1].astype(int)
# #
# #     test = AllPatientsData[AllPatientsData['PatientID'].isin([AllPatientIDs[i] for i in test_indices])].drop(columns=['PatientID']).to_numpy()
# #     test_x = test[:, :-1]
# #     test_y = test[:, -1].astype(int)
# #
# #     clf.fit(train_x, train_y)
# #
# #     lst = clf.predict(test_x)
# #     lst[lst == 1] = 0
# #     lst[lst == -1] = 1
# #
# #     Accuracy.insert(len(Accuracy), accuracy_score(test_y, lst) * 100)
# #     Precision.insert(len(Precision), precision_score(test_y, lst))
# #     Recall.insert(len(Recall), recall_score(test_y, lst))
# #     F1.insert(len(F1), f1_score(test_y, lst))
# #
# #     results.write(
# #         str(cv) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
# #             recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
# #
# #     cv += 1
# #
# # results.write(
# #     'Average,' + str(np.mean(Accuracy)) + ',' + str(np.mean(Precision)) + ',' + str(np.mean(Recall)) + ',' + str(
# #         np.mean(F1)) + '\n')
# # results.close()
# ######################################################################################################################################
# # Most
# os.makedirs(os.path.join(output_path, 'Most'), exist_ok=True)
# results = open(os.path.join(output_path, 'Most', 'Results.csv'), 'w')
# results.write('Cross-Validation,Accuracy,Precision,Recall,F1\n')
#
# cv = 0
# kf = KFold(n_splits=5)
# Accuracy = []
# Precision = []
# Recall = []
# F1 = []
#
# # train = AllPatientsData[MoreVulnerablePatientIDs]
# train = AllPatientsData[AllPatientsData['PatientID'].isin(MoreVulnerablePatientIDs)].drop(columns=['PatientID']).to_numpy()
# train_x = train[:, :-1]
# train_y = train[:, -1].astype(int)
#
# clf.fit(train_x, train_y)
#
# for train_indices, test_indices in kf.split(AllPatientIDs):
#     print('More\tCV: ' + str(cv))
#     test = AllPatientsData[AllPatientsData['PatientID'].isin([AllPatientIDs[i] for i in test_indices])].drop(columns=['PatientID']).to_numpy()
#     test_x = test[:, :-1]
#     test_y = test[:, -1].astype(int)
#
#     lst = clf.predict(test_x)
#     lst[lst == 1] = 0
#     lst[lst == -1] = 1
#
#     Accuracy.insert(len(Accuracy), accuracy_score(test_y, lst) * 100)
#     Precision.insert(len(Precision), precision_score(test_y, lst))
#     Recall.insert(len(Recall), recall_score(test_y, lst))
#     F1.insert(len(F1), f1_score(test_y, lst))
#
#     results.write(
#         str(cv) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
#             recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
#
#     cv += 1
#
# results.write(
#     'Average,' + str(np.mean(Accuracy)) + ',' + str(np.mean(Precision)) + ',' + str(np.mean(Recall)) + ',' + str(
#         np.mean(F1)) + '\n')
# results.close()
# ######################################################################################################################################
# # Least
# os.makedirs(os.path.join(output_path, 'Least'), exist_ok=True)
# results = open(os.path.join(output_path, 'Least', 'Results.csv'), 'w')
# results.write('Cross-Validation,Accuracy,Precision,Recall,F1\n')
#
# cv = 0
# kf = KFold(n_splits=5)
# Accuracy = []
# Precision = []
# Recall = []
# F1 = []
#
# train = AllPatientsData[AllPatientsData['PatientID'].isin(LessVulnerablePatientIDs)].drop(columns=['PatientID']).to_numpy()
# train_x = train[:, :-1]
# train_y = train[:, -1].astype(int)
#
# clf.fit(train_x, train_y)
#
# for train_indices, test_indices in kf.split(AllPatientIDs):
#     print('Less\tCV: ' + str(cv))
#     test = AllPatientsData[AllPatientsData['PatientID'].isin([AllPatientIDs[i] for i in test_indices])].drop(columns=['PatientID']).to_numpy()
#     test_x = test[:, :-1]
#     test_y = test[:, -1].astype(int)
#
#     lst = clf.predict(test_x)
#     lst[lst == 1] = 0
#     lst[lst == -1] = 1
#
#     Accuracy.insert(len(Accuracy), accuracy_score(test_y, lst) * 100)
#     Precision.insert(len(Precision), precision_score(test_y, lst))
#     Recall.insert(len(Recall), recall_score(test_y, lst))
#     F1.insert(len(F1), f1_score(test_y, lst))
#
#     results.write(
#         str(cv) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
#             recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
#
#     cv += 1
#
# results.write(
#     'Average,' + str(np.mean(Accuracy)) + ',' + str(np.mean(Precision)) + ',' + str(np.mean(Recall)) + ',' + str(
#         np.mean(F1)) + '\n')
# results.close()
# ######################################################################################################################################
