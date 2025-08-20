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
# output_path = os.path.join(base_dir, 'Defenses', 'ResultsKNN')
# data_path = os.path.join(base_dir, 'Defenses', 'Data')

training_sets = ['training_setA', 'training_setB']
# os.makedirs(output_path, exist_ok=True)


# get feature names
f = open(os.path.join(base_dir,'inputs/training_setA/p000001.psv'), 'r')
header = f.readline().strip()
features = np.array(header.split('|')[:-1])
f.close()

benign_data_df = pd.DataFrame()
adversarial_data_df = pd.DataFrame()
# adversarial_predictions_df = pd.DataFrame()

i=0
for training_set in training_sets:
    benign_data_path = os.path.join(base_dir, 'results', 'attack_outputs', training_set, 'Data', 'Benign')
    adversarial_data_path = os.path.join(base_dir, 'results', 'attack_outputs', training_set, 'Data', 'Adversarial')
    # adversarial_predictions_path = os.path.join(base_dir, 'results', 'attack_outputs', training_set, 'Predictions', 'Adversarial')
    for f in tqdm(os.listdir(benign_data_path)):
        if os.path.isfile(os.path.join(benign_data_path, f)) and not f.lower().startswith('.') and f.lower().endswith('pkl'):
            benign_data = joblib.load(os.path.join(benign_data_path, f))
            per_patient_df = pd.DataFrame(benign_data)
            per_patient_df.columns = features
            per_patient_df.insert(loc=0, column='PatientID', value=f[:-4])
            benign_data_df = pd.concat([benign_data_df, per_patient_df], axis=0, ignore_index=True)
        else:
            raise Exception('Benign data file does not exist!')

        if os.path.isfile(os.path.join(adversarial_data_path, f)) and not f.lower().startswith('.') and f.lower().endswith('pkl'):
            adversarial_data = joblib.load(os.path.join(adversarial_data_path, f))
            per_patient_df = pd.DataFrame(adversarial_data)
            per_patient_df.columns = features
            per_patient_df.insert(loc=0, column='PatientID', value=f[:-4])
            adversarial_data_df = pd.concat([adversarial_data_df, per_patient_df], axis=0, ignore_index=True)
        else:
            raise Exception('Adversarial data file does not exist!')

        # predictions_f = f[:-4] + '.psv'
        # if os.path.isfile(os.path.join(adversarial_predictions_path, predictions_f)) and not predictions_f.lower().startswith('.') and predictions_f.lower().endswith('psv'):
        #     adversarial_predictions_file = open(os.path.join(adversarial_predictions_path, predictions_f), 'r')
        #     header = adversarial_predictions_file.readline().strip()
        #     per_patient_df = pd.DataFrame(np.loadtxt(adversarial_predictions_file, delimiter='|')[:, 1])
        #     adversarial_predictions_df = pd.concat([adversarial_predictions_df, per_patient_df], axis=0, ignore_index=True)
        # else:
        #     raise Exception('Adversarial prediction file does not exist!')

        if i==1:
            break
        i+=1
    break

# pre-processing
PatientIDs = benign_data_df['PatientID']
selected_features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender']
benign_data_df = benign_data_df.ffill()
benign_data_df = benign_data_df.fillna(0)

benign_data = benign_data_df.loc[:, selected_features]
adversarial_data = adversarial_data_df.loc[:, selected_features]
# adversarial_output = adversarial_predictions_df


# Compare row by row
row_diff = (benign_data != adversarial_data).any(axis=1)

# Insert row_diff into df2 at the last column position
adversarial_data.insert(len(adversarial_data.columns), "Adversarial", row_diff)

print(benign_data)

print(adversarial_data)





#
# neigh = KNeighborsClassifier(n_neighbors=3)
#
# AllPatientsData = joblib.load(data_path + '/AllPatientsData.pkl')
# AllPatientIDs = joblib.load(data_path + '/AllPatientIDs.pkl')
# MostVulnerablePatientIDs = joblib.load(data_path + '/MostVulnerablePatientIDs.pkl')
# MoreVulnerablePatientIDs = joblib.load(data_path + '/MoreVulnerablePatientIDs.pkl')
# LessVulnerablePatientIDs = joblib.load(data_path + '/LessVulnerablePatientIDs.pkl')
#
# ######################################################################################################################################
# # Samples
# os.system('mkdir ' + base_dir + '/ResultsKNN/SamplesTraining')
# results = open(output_path + '/SamplesTraining/Results.csv', 'w')
# results.write('Run,Accuracy,Precision,Recall,F1\n')
#
# Accuracy = []
# Precision = []
# Recall = []
# F1 = []
# for run in range(5):
#     print('SamplesTraining\tTrial: ' + str(run))
#
#     split = train_test_split(AllPatientIDs, train_size=len(LessVulnerablePatientIDs))
#     train_indices = split[0]
#     test_indices = split[1][:math.floor(0.2 * len(AllPatientIDs))]
#
#     train = AllPatientsData[train_indices]
#     train = train.reshape(-1, train.shape[2])
#     train_x = train[:, :-1]
#     train_y = train[:, -1]
#
#     test = AllPatientsData[test_indices]
#     test = test.reshape(-1, test.shape[2])
#     test_x = test[:, :-1]
#     test_y = test[:, -1]
#
#     neigh.fit(train_x, train_y)
#
#     lst = neigh.predict(test_x)
#
#     Accuracy.insert(len(Accuracy), accuracy_score(test_y, lst) * 100)
#     Precision.insert(len(Precision), precision_score(test_y, lst))
#     Recall.insert(len(Recall), recall_score(test_y, lst))
#     F1.insert(len(F1), f1_score(test_y, lst))
#
#     results.write(
#         str(run) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
#             recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
#
# results.write(
#     'Average,' + str(np.mean(Accuracy)) + ',' + str(np.mean(Precision)) + ',' + str(np.mean(Recall)) + ',' + str(
#         np.mean(F1)) + '\n')
# results.close()
# ######################################################################################################################################
# # All patients
# os.system('mkdir ' + base_dir + '/ResultsKNN/All')
# results = open(output_path + '/All/Results.csv', 'w')
# results.write('Cross-Validation,Accuracy,Precision,Recall,F1\n')
#
# cv = 0
# kf = KFold(n_splits=5)
# Accuracy = []
# Precision = []
# Recall = []
# F1 = []
# for train_indices, test_indices in kf.split(AllPatientIDs):
#     print('All\tCV: ' + str(cv))
#
#     train = AllPatientsData[train_indices]
#     train = train.reshape(-1, train.shape[2])
#     train_x = train[:, :-1]
#     train_y = train[:, -1]
#
#     test = AllPatientsData[test_indices]
#     test = test.reshape(-1, test.shape[2])
#     test_x = test[:, :-1]
#     test_y = test[:, -1]
#
#     neigh.fit(train_x, train_y)
#
#     lst = neigh.predict(test_x)
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
# # Most
# os.system('mkdir ' + base_dir + '/ResultsKNN/Most')
# results = open(output_path + '/Most/Results.csv', 'w')
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
# train = AllPatientsData[random.sample(MoreVulnerablePatientIDs, 1000)]
# train = train.reshape(-1, train.shape[2])
# train_x = train[:, :-1]
# train_y = train[:, -1]
#
# neigh.fit(train_x, train_y)
#
# for train_indices, test_indices in kf.split(AllPatientIDs):
#     print('More\tCV: ' + str(cv))
#     test = AllPatientsData[test_indices]
#     test = test.reshape(-1, test.shape[2])
#     test_x = test[:, :-1]
#     test_y = test[:, -1]
#
#     lst = neigh.predict(test_x)
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
# os.system('mkdir ' + base_dir + '/ResultsKNN/Least')
# results = open(output_path + '/Least/Results.csv', 'w')
# results.write('Cross-Validation,Accuracy,Precision,Recall,F1\n')
#
# cv = 0
# kf = KFold(n_splits=5)
# Accuracy = []
# Precision = []
# Recall = []
# F1 = []
#
# train = AllPatientsData[LessVulnerablePatientIDs]
# train = train.reshape(-1, train.shape[2])
# train_x = train[:, :-1]
# train_y = train[:, -1]
#
# neigh.fit(train_x, train_y)
#
# for train_indices, test_indices in kf.split(AllPatientIDs):
#     print('Less\tCV: ' + str(cv))
#     test = AllPatientsData[test_indices]
#     test = test.reshape(-1, test.shape[2])
#     test_x = test[:, :-1]
#     test_y = test[:, -1]
#
#     lst = neigh.predict(test_x)
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
