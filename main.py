import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
import joblib
import time
import numpy as np

base_dir = '/home/mnawawy/Downloads/OhioT1DM/processed_data'
# base_dir = '/Users/nawawy/Desktop/Research/OhioT1DM_data'
data_dir = os.path.join(base_dir, 'training_subsets')

neigh = KNeighborsClassifier(n_neighbors=7)
clf = OneClassSVM(gamma='auto')#, kernel='sigmoid', coef0=10)

print('kNN')
######################################################################################################################################
# All patients
train = np.load(data_dir+'/ohiot1dm_train_all_0.npy')
train_x = train[:, :-1]
train_y = train[:, -1]

start_time = time.perf_counter()
neigh.fit(train_x, train_y)
end_time = time.perf_counter()
elapsed_time_all = end_time - start_time
print(f"All Patients Elapsed Time: {elapsed_time_all:.6f} seconds")
######################################################################################################################################
# Least
train = np.load(data_dir+'/ohiot1dm_train_least_0.npy')
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
train = np.load(data_dir+'/ohiot1dm_train_all_0.npy')
train_x = train[:, :-1]
train_y = train[:, -1]
start_time = time.perf_counter()
clf.fit(train_x)
end_time = time.perf_counter()
elapsed_time_all = end_time - start_time
print(f"All Patients Elapsed Time: {elapsed_time_all:.6f} seconds")
######################################################################################################################################
# Least
train = np.load(data_dir+'/ohiot1dm_train_least_0.npy')
train_x = train[:, :-1]
train_y = train[:, -1]
start_time = time.perf_counter()
clf.fit(train_x)
end_time = time.perf_counter()
elapsed_time_less = end_time - start_time
print(f"Less Vulnerable Elapsed Time: {elapsed_time_less:.6f} seconds")

######################################################################################################################################
print('Percentage Decrease One-Class SVM = '+str(((elapsed_time_all-elapsed_time_less)/elapsed_time_all)*100))