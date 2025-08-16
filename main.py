import csv
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
import os

import warnings
warnings.filterwarnings("ignore")

base_dir = '/Users/nawawy/Desktop/Research/Sepsis_data/outputs'
training_sets = ['training_setA', 'training_setB']

# benign_files = []
adversarial_files = []

# get features
columns_file = open('/Users/nawawy/Desktop/Research/Sepsis_data/physionet.org/files/challenge-2019/1.0.0/training/training_setA/p000001.psv', 'r')
header = columns_file.readline().strip()
features = np.array(header.split('|')[:-1])
columns_file.close()


# benign_df = pd.DataFrame()
adversarial_df = pd.DataFrame()

for training_set in training_sets:
    # benign_path = os.path.join(base_dir, training_set, 'Data', 'Benign')
    adversarial_path = os.path.join(base_dir, training_set, 'Data', 'Adversarial')
    for f in os.listdir(adversarial_path):
        # if os.path.isfile(os.path.join(benign_path, f)) and not f.lower().startswith('.') and f.lower().endswith('pkl'):
        #     benign_data = joblib.load(os.path.join(benign_path, f))
        #     benign_patient_df = pd.DataFrame(benign_data)
        #     benign_patient_df.columns = features
        #     benign_patient_df.insert(loc=0, column='PatientID', value=f[:-4])
        #     benign_df = pd.concat([benign_df, benign_patient_df], axis=0, ignore_index=True)
        # else:
        #     raise Exception('Benign data file does not exist!')

        if os.path.isfile(os.path.join(adversarial_path, f)) and not f.lower().startswith('.') and f.lower().endswith('pkl'):
            adversarial_data = joblib.load(os.path.join(adversarial_path, f))
            adversarial_patient_df = pd.DataFrame(adversarial_data)
            adversarial_patient_df.columns = features
            adversarial_patient_df.insert(loc=0, column='PatientID', value=f[:-4])
            adversarial_df = pd.concat([adversarial_df, adversarial_patient_df], axis=0, ignore_index=True)
        else:
            raise Exception('Adversarial data file does not exist!')


# adversarial_data = joblib.load('/Users/nawawy/Desktop/Research/Old/MortalityData/True/adversarial_data.pkl')
# adversarial_output = joblib.load('/Users/nawawy/Desktop/Research/Old/MortalityData/True/adversarial_output.pkl')
# benign_output = joblib.load('/Users/nawawy/Desktop/Research/Old/MortalityData/False/benign_output.pkl')
# print('adversarial_data')
# print(adversarial_data)
# print(adversarial_data.shape)
# print(type(adversarial_data))
# print('---------------------------------------------')


# most_vulnerable = []
# for i in range(len(benign_output)):
#     if adversarial_output[i] >= 0.5 and benign_output[i] < 0.5:
#         most_vulnerable.append(i)
#
# joblib.dump(most_vulnerable, './MostVulnerablePatientIDs.pkl')


features = np.insert(features,0,"PatientID")

# pre-processing
# adversarial_data = adversarial_data.reshape(-1, 72 * 432)
# print('adversarial_data')
# print(adversarial_data)
# print(adversarial_data.shape)
# print(type(adversarial_data))
# print('---------------------------------------------')
# exit(1)
adversarial_df = preprocessing.normalize(adversarial_df)
print(adversarial_df)
exit(1)
adversarial_data = preprocessing.normalize(adversarial_data)
adversarial_output = adversarial_output >= 0.5

# logistic regression for feature importance
# define dataset
X = adversarial_data
y = adversarial_output
# define the model
model = LogisticRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]

timeseries = importance*adversarial_data
timeseries = timeseries.reshape(-1, 72, 432)

risk_profiles = []
features = [143, 83, 89, 118, 183, 114, 177, 47, 75, 100]

for i in range(len(timeseries)):
    df = pd.DataFrame(timeseries[i])
    risk_profiles.append(df[features].to_numpy().reshape(72*len(features)))

df = pd.DataFrame(risk_profiles)
joblib.dump(df, 'RiskProfiles.pkl')
model = KMeans(n_clusters=2)
model.fit(df)
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

if countA > countB:
    joblib.dump(clusterA, './MoreVulnerablePatientIDs.pkl')
    joblib.dump(clusterB, './LessVulnerablePatientIDs.pkl')
else:
    joblib.dump(clusterA, './LessVulnerablePatientIDs.pkl')
    joblib.dump(clusterB, './MoreVulnerablePatientIDs.pkl')
joblib.dump(list(range(0,len(predictions))), './AllPatientIDs.pkl')
