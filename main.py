import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
import joblib
import time
import numpy as np

base_dir = '/home/mnawawy/Sepsis'
# base_dir = '/Users/nawawy/Desktop/Research/Sepsis_data'

AllPatientsData = joblib.load(os.path.join(base_dir, 'results', 'attack_outputs', 'adversarial_data.pkl'))

neigh = KNeighborsClassifier(n_neighbors=3)
clf = OneClassSVM(gamma='scale', kernel='linear', verbose=True)

AllPatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'AllPatientIDs.pkl'))
MostVulnerablePatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'threshold_10', 'MostVulnerablePatientIDs.pkl'))
MoreVulnerablePatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'threshold_5', 'MoreVulnerablePatientIDs.pkl'))
LessVulnerablePatientIDs = joblib.load(os.path.join(base_dir, 'results', 'cluster_outputs', 'benign_data_adv_outputs', 'threshold_5', 'LessVulnerablePatientIDs.pkl'))



print('kNN')
######################################################################################################################################
# All patients
train = AllPatientsData.drop(columns=['PatientID']).to_numpy()
train_x = train[:, :-1]
train_y = train[:, -1].astype(int)

start_time = time.perf_counter()
neigh.fit(train_x, train_y)
end_time = time.perf_counter()
elapsed_time_all = end_time - start_time
print(f"All Patients Elapsed Time: {elapsed_time_all:.6f} seconds")
######################################################################################################################################
# Least
train = AllPatientsData[AllPatientsData['PatientID'].isin(LessVulnerablePatientIDs)].drop(columns=['PatientID']).to_numpy()
train_x = train[:, :-1]
train_y = train[:, -1].astype(int)
start_time = time.perf_counter()
neigh.fit(train_x, train_y)
end_time = time.perf_counter()
elapsed_time_less = end_time - start_time
print(f"Less Vulnerable Elapsed Time: {elapsed_time_less:.6f} seconds")

######################################################################################################################################
print('Percentage Decrease kNN = '+str(((elapsed_time_all-elapsed_time_less)/elapsed_time_all)*100))
print('------------------------------------------------------')
print('One-Class SVM')
######################################################################################################################################
# All patients
train = AllPatientsData.drop(columns=['PatientID']).to_numpy()
train_x = train[:, :-1]
train_y = train[:, -1].astype(int)
start_time = time.perf_counter()
clf.fit(train_x, train_y)
end_time = time.perf_counter()
elapsed_time_all = end_time - start_time
print(f"All Patients Elapsed Time: {elapsed_time_all:.6f} seconds")
######################################################################################################################################
# Least
train = AllPatientsData[AllPatientsData['PatientID'].isin(LessVulnerablePatientIDs)].drop(columns=['PatientID']).to_numpy()
train_x = train[:, :-1]
train_y = train[:, -1].astype(int)
start_time = time.perf_counter()
clf.fit(train_x, train_y)
end_time = time.perf_counter()
elapsed_time_less = end_time - start_time
print(f"Less Vulnerable Elapsed Time: {elapsed_time_less:.6f} seconds")

######################################################################################################################################
print('Percentage Decrease One-Class SVM = '+str(((elapsed_time_all-elapsed_time_less)/elapsed_time_all)*100))