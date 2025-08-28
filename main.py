import csv
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans

import warnings
import os
warnings.filterwarnings("ignore")

base_dir = '/home/mnawawy'

most_vulnerable = joblib.load(os.path.join(base_dir, 'MortalityData/Data/MostVulnerablePatientIDs.pkl'))

df = joblib.load(os.path.join(base_dir, 'MortalityData/Data/RiskProfiles.pkl'))
model = KMeans(n_clusters=2)
model.fit(df)

cluster_centers = model.cluster_centers_
print(cluster_centers)

from sklearn.metrics.pairwise import euclidean_distances

# Calculate distances between all pairs of cluster centers
distances_matrix = euclidean_distances(cluster_centers)
print("Distance matrix between cluster centers:\n", distances_matrix)

predictions = model.predict(df)

clusterA = []
clusterB = []
for i in range(len(predictions)):
    if predictions[i] == 0:
        clusterA.append(i)
    else:
        clusterB.append(i)

countA = 0
countB = 0
for i in range(len(most_vulnerable)):
    if most_vulnerable[i] in clusterA:
        countA += 1
    elif most_vulnerable[i] in clusterB:
        countB += 1


print('Cluster A Patients: '+str(len(clusterA)))
print('Cluster B Patients: '+str(len(clusterB)))
print('Most Vulnerable in Cluster A: '+str(countA))
print('Most Vulnerable in Cluster B: '+str(countB))
print('Percentage of Most Vulnerable in Cluster A: '+ str((countA/(countA+countB))*100))
print('Percentage of Most Vulnerable in Cluster B: '+ str((countB/(countA+countB))*100))



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
