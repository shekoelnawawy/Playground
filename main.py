import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import joblib
import random
import warnings
import math
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

base_dir = '/home/mnawawy/Sepsis'
# base_dir = '/Users/nawawy/Desktop/Research/Sepsis_data'
output_path = os.path.join(base_dir, 'Defenses', 'ResultsKNN')


AllPatientsData = joblib.load(os.path.join(base_dir, 'results', 'attack_outputs', 'adversarial_data.pkl'))

neigh = KNeighborsClassifier(n_neighbors=3)


AllPatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'AllPatientIDs.pkl'))
MostVulnerablePatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'threshold_5', 'MostVulnerablePatientIDs.pkl'))
MoreVulnerablePatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'threshold_5', 'MoreVulnerablePatientIDs.pkl'))
LessVulnerablePatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'threshold_5', 'LessVulnerablePatientIDs.pkl'))

######################################################################################################################################
# Samples
os.makedirs(os.path.join(output_path, 'SamplesTraining'), exist_ok=True)
results = open(os.path.join(output_path, 'SamplesTraining', 'Results.csv'), 'w')
results.write('Run,Accuracy,Precision,Recall,F1\n')

Accuracy = []
Precision = []
Recall = []
F1 = []
for run in range(5):
    print('SamplesTraining\tTrial: ' + str(run))

    split = train_test_split(AllPatientIDs, train_size=len(LessVulnerablePatientIDs))
    train_indices = split[0]
    test_indices = split[1][:math.floor(0.2 * len(AllPatientIDs))]

    train = AllPatientsData[AllPatientsData['PatientID'].isin(train_indices)].drop(columns=['PatientID']).to_numpy()
    train_x = train[:, :-1]
    train_y = train[:, -1].astype(int)

    test = AllPatientsData[AllPatientsData['PatientID'].isin(test_indices)].drop(columns=['PatientID']).to_numpy()
    test_x = test[:, :-1]
    test_y = test[:, -1].astype(int)

    neigh.fit(train_x, train_y)

    lst = neigh.predict(test_x)

    Accuracy.insert(len(Accuracy), accuracy_score(test_y, lst) * 100)
    Precision.insert(len(Precision), precision_score(test_y, lst))
    Recall.insert(len(Recall), recall_score(test_y, lst))
    F1.insert(len(F1), f1_score(test_y, lst))

    results.write(
        str(run) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
            recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')

results.write(
    'Average,' + str(np.mean(Accuracy)) + ',' + str(np.mean(Precision)) + ',' + str(np.mean(Recall)) + ',' + str(
        np.mean(F1)) + '\n')
results.close()
######################################################################################################################################
# All patients
os.makedirs(os.path.join(output_path, 'All'), exist_ok=True)
results = open(os.path.join(output_path, 'All', 'Results.csv'), 'w')
results.write('Cross-Validation,Accuracy,Precision,Recall,F1\n')

cv = 0
kf = KFold(n_splits=5)
Accuracy = []
Precision = []
Recall = []
F1 = []
for train_indices, test_indices in kf.split(AllPatientIDs):
    print('All\tCV: ' + str(cv))

    train = AllPatientsData[AllPatientsData['PatientID'].isin(train_indices)].drop(columns=['PatientID']).to_numpy()
    train_x = train[:, :-1]
    train_y = train[:, -1].astype(int)

    test = AllPatientsData[AllPatientsData['PatientID'].isin(test_indices)].drop(columns=['PatientID']).to_numpy()
    test_x = test[:, :-1]
    test_y = test[:, -1].astype(int)

    neigh.fit(train_x, train_y)

    lst = neigh.predict(test_x)

    Accuracy.insert(len(Accuracy), accuracy_score(test_y, lst) * 100)
    Precision.insert(len(Precision), precision_score(test_y, lst))
    Recall.insert(len(Recall), recall_score(test_y, lst))
    F1.insert(len(F1), f1_score(test_y, lst))

    results.write(
        str(cv) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
            recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')

    cv += 1

results.write(
    'Average,' + str(np.mean(Accuracy)) + ',' + str(np.mean(Precision)) + ',' + str(np.mean(Recall)) + ',' + str(
        np.mean(F1)) + '\n')
results.close()
######################################################################################################################################
# Most
os.makedirs(os.path.join(output_path, 'Most'), exist_ok=True)
results = open(os.path.join(output_path, 'Most', 'Results.csv'), 'w')
results.write('Cross-Validation,Accuracy,Precision,Recall,F1\n')

cv = 0
kf = KFold(n_splits=5)
Accuracy = []
Precision = []
Recall = []
F1 = []

# train = AllPatientsData[MoreVulnerablePatientIDs]
train = AllPatientsData[AllPatientsData['PatientID'].isin(random.sample(MoreVulnerablePatientIDs, 1000))].drop(columns=['PatientID']).to_numpy()
train_x = train[:, :-1]
train_y = train[:, -1].astype(int)

neigh.fit(train_x, train_y)

for train_indices, test_indices in kf.split(AllPatientIDs):
    print('More\tCV: ' + str(cv))
    test = AllPatientsData[AllPatientsData['PatientID'].isin(test_indices)].drop(columns=['PatientID']).to_numpy()
    test_x = test[:, :-1]
    test_y = test[:, -1].astype(int)

    lst = neigh.predict(test_x)

    Accuracy.insert(len(Accuracy), accuracy_score(test_y, lst) * 100)
    Precision.insert(len(Precision), precision_score(test_y, lst))
    Recall.insert(len(Recall), recall_score(test_y, lst))
    F1.insert(len(F1), f1_score(test_y, lst))

    results.write(
        str(cv) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
            recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')

    cv += 1

results.write(
    'Average,' + str(np.mean(Accuracy)) + ',' + str(np.mean(Precision)) + ',' + str(np.mean(Recall)) + ',' + str(
        np.mean(F1)) + '\n')
results.close()
######################################################################################################################################
# Least
os.makedirs(os.path.join(output_path, 'Least'), exist_ok=True)
results = open(os.path.join(output_path, 'Least', 'Results.csv'), 'w')
results.write('Cross-Validation,Accuracy,Precision,Recall,F1\n')

cv = 0
kf = KFold(n_splits=5)
Accuracy = []
Precision = []
Recall = []
F1 = []

train = AllPatientsData[AllPatientsData['PatientID'].isin(LessVulnerablePatientIDs)].drop(columns=['PatientID']).to_numpy()
train_x = train[:, :-1]
train_y = train[:, -1].astype(int)

neigh.fit(train_x, train_y)

for train_indices, test_indices in kf.split(AllPatientIDs):
    print('Less\tCV: ' + str(cv))
    test = AllPatientsData[AllPatientsData['PatientID'].isin(test_indices)].drop(columns=['PatientID']).to_numpy()
    test_x = test[:, :-1]
    test_y = test[:, -1].astype(int)

    lst = neigh.predict(test_x)

    Accuracy.insert(len(Accuracy), accuracy_score(test_y, lst) * 100)
    Precision.insert(len(Precision), precision_score(test_y, lst))
    Recall.insert(len(Recall), recall_score(test_y, lst))
    F1.insert(len(F1), f1_score(test_y, lst))

    results.write(
        str(cv) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
            recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')

    cv += 1

results.write(
    'Average,' + str(np.mean(Accuracy)) + ',' + str(np.mean(Precision)) + ',' + str(np.mean(Recall)) + ',' + str(
        np.mean(F1)) + '\n')
results.close()
######################################################################################################################################
