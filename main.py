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
# Least
train = AllPatientsData[AllPatientsData['PatientID'].isin(LessVulnerablePatientIDs)].drop(columns=['PatientID']).to_numpy()
train_x = train[:, :-1]
train_y = train[:, -1].astype(int)
start_train_time = time.perf_counter()
neigh.fit(train_x, train_y)
end_train_time = time.perf_counter()
elapsed_train_time = end_train_time - start_train_time
print(f"Total Training Time: {elapsed_train_time:.6f} seconds")

test = AllPatientsData[AllPatientsData['PatientID'].isin(AllPatientIDs)].drop(columns=['PatientID']).to_numpy()
test_x = test[:, :-1]
test_y = test[:, -1].astype(int)

start_test_time = time.perf_counter()
lst = neigh.predict(test_x)
end_test_time = time.perf_counter()
elapsed_test_time = end_test_time - start_test_time
print(f"Inference Time per instance: {elapsed_test_time/len(test):.6f} seconds")
######################################################################################################################################

print('------------------------------------------------------')
print('One-Class SVM')
######################################################################################################################################
# Least
train = AllPatientsData[AllPatientsData['PatientID'].isin(LessVulnerablePatientIDs)].drop(columns=['PatientID']).to_numpy()
train_x = train[:, :-1]
train_y = train[:, -1].astype(int)
start_train_time = time.perf_counter()
clf.fit(train_x, train_y)
end_train_time = time.perf_counter()
elapsed_train_time = end_train_time - start_train_time
print(f"Total Training Time: {elapsed_train_time:.6f} seconds")

test = AllPatientsData[AllPatientsData['PatientID'].isin(AllPatientIDs)].drop(columns=['PatientID']).to_numpy()
test_x = test[:, :-1]
test_y = test[:, -1].astype(int)

start_test_time = time.perf_counter()
lst = clf.predict(test_x)
end_test_time = time.perf_counter()
elapsed_test_time = end_test_time - start_test_time
print(f"Inference Time per instance: {elapsed_test_time/len(test):.6f} seconds")
lst[lst == 1] = 0
lst[lst == -1] = 1

######################################################################################################################################
