import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
import joblib
import time

base_dir = '/home/mnawawy/Downloads/MIMIC/processed_data/risk_profiling'
# base_dir = '/Users/nawawy/Desktop/Research/MIMIC_data'
data_dir = os.path.join(base_dir, 'DataOld')

neigh = KNeighborsClassifier(n_neighbors=3)
clf = OneClassSVM(gamma='scale', kernel='linear', verbose=True)

AllPatientsData = joblib.load(data_dir + '/AllPatientsData.pkl')
AllPatientIDs = joblib.load(data_dir + '/AllPatientIDs.pkl')
MostVulnerablePatientIDs = joblib.load(data_dir + '/MostVulnerablePatientIDs.pkl')
MoreVulnerablePatientIDs = joblib.load(data_dir + '/MoreVulnerablePatientIDs.pkl')
LessVulnerablePatientIDs = joblib.load(data_dir + '/LessVulnerablePatientIDs.pkl')

print('Percentage Decrease training set = '+str(((len(AllPatientIDs)-len(LessVulnerablePatientIDs))/len(AllPatientIDs))*100))
print('------------------------------------------------------')
print('kNN')
######################################################################################################################################
# All patients
train = AllPatientsData[AllPatientIDs]
train = train.reshape(-1, train.shape[2])
train_x = train[:, :-1]
train_y = train[:, -1]
start_time = time.perf_counter()
neigh.fit(train_x, train_y)
end_time = time.perf_counter()
elapsed_time_all = end_time - start_time
print(f"All Patients Elapsed Time: {elapsed_time_all:.6f} seconds")
######################################################################################################################################
# Least
train = AllPatientsData[LessVulnerablePatientIDs]
train = train.reshape(-1, train.shape[2])
train_x = train[:, :-1]
train_y = train[:, -1]
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
train = AllPatientsData[AllPatientIDs]
train = train.reshape(-1, train.shape[2])
train_x = train[:, :-1]
train_y = train[:, -1]
start_time = time.perf_counter()
clf.fit(train_x, train_y)
end_time = time.perf_counter()
elapsed_time_all = end_time - start_time
print(f"All Patients Elapsed Time: {elapsed_time_all:.6f} seconds")
######################################################################################################################################
# Least
train = AllPatientsData[LessVulnerablePatientIDs]
train = train.reshape(-1, train.shape[2])
train_x = train[:, :-1]
train_y = train[:, -1]
start_time = time.perf_counter()
clf.fit(train_x, train_y)
end_time = time.perf_counter()
elapsed_time_less = end_time - start_time
print(f"Less Vulnerable Elapsed Time: {elapsed_time_less:.6f} seconds")

######################################################################################################################################
print('Percentage Decrease kNN = '+str(((elapsed_time_all-elapsed_time_less)/elapsed_time_all)*100))