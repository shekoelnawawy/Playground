from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
import os
import warnings

warnings.filterwarnings("ignore")

most_vulnerable_threshold = 25

# base_dir = '/Users/nawawy/Desktop/Research/Sepsis_data'
base_dir = '/home/mnawawy/Sepsis'
training_sets = ['training_setA', 'training_setB']
os.makedirs(os.path.join(base_dir,'cluster_outputs'), exist_ok=True)

# get feature names
f = open(os.path.join(base_dir,'inputs/training_setA/p000001.psv'), 'r')
header = f.readline().strip()
features = np.array(header.split('|')[:-1])
f.close()

adversarial_data_df = pd.DataFrame()
adversarial_predictions_df = pd.DataFrame()
i=0
for training_set in training_sets:
    adversarial_data_path = os.path.join(base_dir, 'outputs', training_set, 'Data', 'Adversarial')
    adversarial_predictions_path = os.path.join(base_dir, 'outputs', training_set, 'Predictions', 'Adversarial')
    for f in os.listdir(adversarial_data_path):
        if os.path.isfile(os.path.join(adversarial_data_path, f)) and not f.lower().startswith('.') and f.lower().endswith('pkl'):
            adversarial_data = joblib.load(os.path.join(adversarial_data_path, f))
            adversarial_patient_df = pd.DataFrame(adversarial_data)
            adversarial_patient_df.columns = features
            adversarial_patient_df.insert(loc=0, column='PatientID', value=f[:-4])
            adversarial_data_df = pd.concat([adversarial_data_df, adversarial_patient_df], axis=0, ignore_index=True)
        else:
            raise Exception('Adversarial data file does not exist!')

        predictions_f = f[:-4] + '.psv'
        if os.path.isfile(os.path.join(adversarial_predictions_path, predictions_f)) and not predictions_f.lower().startswith('.') and predictions_f.lower().endswith('psv'):
            adversarial_predictions_file = open(os.path.join(adversarial_predictions_path, predictions_f), 'r')
            header = adversarial_predictions_file.readline().strip()
            adversarial_patient_df = pd.DataFrame(np.loadtxt(adversarial_predictions_file, delimiter='|')[:, 1])
            adversarial_predictions_df = pd.concat([adversarial_predictions_df, adversarial_patient_df], axis=0, ignore_index=True)
        else:
            raise Exception('Adversarial prediction file does not exist!')
        if i == 1:
            break
        i+=1
    break
# pre-processing
PatientIDs = adversarial_data_df['PatientID']
selected_features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender']
adversarial_data = np.array(adversarial_data_df.loc[:, selected_features])
adversarial_data = preprocessing.normalize(adversarial_data)
adversarial_output = np.array(adversarial_predictions_df)

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

timeseries = abs(importance*adversarial_data)

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

joblib.dump(risk_profiles, os.path.join(base_dir, 'cluster_outputs', 'RiskProfiles.pkl'))

unique_PatientIDs = [item[0] for item in risk_profiles]
df = pd.DataFrame([item[1] for item in risk_profiles]).ffill(axis=1)
model = KMeans(n_clusters=2)
model.fit(df)
predictions = model.predict(df)

mispredictions = pd.read_csv(os.path.join(base_dir, 'outputs/percentage_mispredictions.csv'))
most_vulnerable = mispredictions[mispredictions['PercentageMisprediction']>most_vulnerable_threshold]['PatientID'].tolist()
joblib.dump(most_vulnerable, os.path.join(base_dir, 'cluster_outputs', 'MostVulnerablePatientIDs.pkl'))

clusterA = []
clusterB = []
for i in range(len(predictions)):
    if predictions[i] == 0:
        clusterA.append(unique_PatientIDs[i])
    else:
        clusterB.append(unique_PatientIDs[i])

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
if countA+countB != 0:
    print('Percentage of Most Vulnerable in Cluster A: '+ str((countA/(countA+countB))*100))
    print('Percentage of Most Vulnerable in Cluster B: '+ str((countB/(countA+countB))*100))

if countA > countB:
    joblib.dump(clusterA, os.path.join(base_dir, 'cluster_outputs', 'MoreVulnerablePatientIDs.pkl'))
    joblib.dump(clusterB, os.path.join(base_dir, 'cluster_outputs', 'LessVulnerablePatientIDs.pkl'))
else:
    joblib.dump(clusterA, os.path.join(base_dir, 'cluster_outputs', 'LessVulnerablePatientIDs.pkl'))
    joblib.dump(clusterB, os.path.join(base_dir, 'cluster_outputs', 'MoreVulnerablePatientIDs.pkl'))
joblib.dump(mispredictions['PatientID'].tolist(), os.path.join(base_dir, 'cluster_outputs', 'AllPatientIDs.pkl'))
