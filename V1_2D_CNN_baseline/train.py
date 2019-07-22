'''
CONV1D
channel 1,2,3,4: xxxx xxxx xxxx xxxx
'''

import os
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
import random
import scipy.io as scio

# Hyperparameters
path = "../../.."
xlsx_path = "../../../Monash_University_Seizure_Detection_Database_" \
            "September_2018_Deidentified.xlsx"
sheet_name = "Seizure Information"

eeg_signals = ['C3', 'F7', 'F4', 'C4', 'Fz', 'Cz', 'Pz', 'Fp1',
               'P3', 'Fp2', 'P4', 'F3', 'F8']
ecg_signals = ['ECG']
res_signals = ['THO-', 'THO+', 'Air Flow']

hz = 250 # hertz = 250
window_size = 500
stride = 250
p_n_rate = 1/2 # seizure:non-seizure rate = 1:5 (Totally 9011 seizure window)
train_rate = 0.7 # 70% patients data are used for training model
val_rate = 0.2 # 20% patients data are used for validation
test_rate = 0.1 # 10% patients data are used for testing

# normalization function
# input: array
# output: normalized array
def normalization(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 2e-12)

# function read_file_name
# input: path of the file directory "../../xo13"
# output: file path array under the file directory
def read_file_name(path):
    files = os.listdir(path)
    file_name = []
    for file in files:
        if file[-3:] == "edf":
            file_path = path + "/" + file
            file_name.append(file_path)
    return file_name

# For each patient, read eeg, ecg, res data individual
# Input file path "../../xo13/CPM027.edf"
# Output eeg, ecg, res data
def read_data(data_path, eeg_signals, ecg_signals, res_signals):
    rawData = read_raw_edf(data_path)
    tmp = rawData.to_data_frame()
    eeg_data = tmp[eeg_signals]
    ecg_data = tmp[ecg_signals]
    res_data = tmp[res_signals]
    return eeg_data, ecg_data, res_data

# For xlsx file, read Patient ID, Recording Start, Seizure Start, Seizure End Information
# Input: xlsx file path, sheet name
# Output: Dataframe - "Patient ID", "Recording Start", "Seizure Start", "Seizure End"
def read_csv(xlsx_path, sheet_name = "Seizure Information"):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df = df[["Patient ID", "Recording Start", "Seizure Start", "Seizure End"]]
    df = df[df["Patient ID"].isnull() == False]
    temp = {}
    for _, row in df.iterrows():
        if not pd.isnull(row["Recording Start"]):
            temp[row["Patient ID"]] = row["Recording Start"]
    for _, row in df.iterrows():
        if pd.isnull(row["Recording Start"]) and not pd.isnull(row["Patient ID"]):
            row["Recording Start"] = temp[row["Patient ID"]]
    return df

# function change time to corresponding index
# input: dataframe of time
# output: dataframe of index
def time_to_index(recording, seizure):
    recording = pd.to_timedelta(recording.astype(str))
    seizure = pd.to_timedelta(seizure.astype(str))
    index = ((seizure - recording) / np.timedelta64(1, 's')) * hz
    return index

# function calculate corresponding seizure start / index for each patient
# input: patient ID, dataframe from function - read_csv()
# output: array containing seizure start / end index - [[], [], []]
def generate_seizure_index(patient_id, df):
    df = df[df["Patient ID"] == patient_id]
    seizure = []
    start_index = time_to_index(df["Recording Start"], df["Seizure Start"])
    end_index = time_to_index(df["Recording Start"], df["Seizure End"])
    for start, end in zip(start_index, end_index):
        if start < 0:
            start = start + 24 * 3600 * hz
        if end < 0:
            end = end + 24 * 3600 * hz
        seizure.append([start, end])
    return seizure

def window_gen(eeg_data, ecg_data, res_data, seizure_indexs):
    batch_x_eeg = []  # seizure
    batch_x_ecg = []
    batch_x_res = []
    batch_y = []
    batch_x_eeg_n = []  # non-seizure
    batch_x_ecg_n = []
    batch_x_res_n = []
    batch_y_n = []
    for index in range(0, (len(eeg_data) - window_size), stride):
        y = [1, 0] # where 1,0 stands no_seizure, 0,1 stands for seizure
        for s_arr in seizure_indexs:
            if (index >= s_arr[0]) and ((index + window_size) <= s_arr[1]):
                y = [0, 1]
        if y == [0, 1]:
            batch_y.append(y)
            batch_x_eeg.append(normalization(eeg_data[index: (index + window_size)].values.reshape(1, window_size*len(eeg_data.columns))[0]))
            batch_x_ecg.append(normalization(ecg_data[index: (index + window_size)].values.reshape(1, window_size*len(ecg_data.columns))[0]))
            batch_x_res.append(normalization(res_data[index: (index + window_size)].values.reshape(1, window_size * len(res_data.columns))[0]))
        else:
            batch_y_n.append(y)
            batch_x_eeg_n.append(normalization(eeg_data[index: (index + window_size)].values.reshape(1, window_size * len(eeg_data.columns))[0]))
            batch_x_ecg_n.append(normalization(ecg_data[index: (index + window_size)].values.reshape(1, window_size * len(ecg_data.columns))[0]))
            batch_x_res_n.append(normalization(res_data[index: (index + window_size)].values.reshape(1, window_size * len(res_data.columns))[0]))
    index = np.random.randint(0, len(batch_x_eeg_n), int(len(batch_x_eeg) / p_n_rate))
    batch_y += [batch_y_n[j] for j in index]
    batch_x_eeg += [batch_x_eeg_n[j] for j in index]
    batch_x_ecg += [batch_x_ecg_n[j] for j in index]
    batch_x_res += [batch_x_res_n[j] for j in index]
    return batch_x_eeg, batch_x_ecg, batch_x_res, batch_y

def xy_gen(path, xlsx_path, sheet_name = "Seizure Information"):
    df = read_csv(xlsx_path, sheet_name)
    file_path = read_file_name(path)
    training_set = {"x_eeg":[], "x_ecg":[], "x_res":[], "y":[]} # training set
    validation_set = {"x_eeg": [], "x_ecg": [], "x_res": [], "y": []} # validation set
    test_set = {"x_eeg": [], "x_ecg": [], "x_res": [], "y": []} # test set
    training_flag = random.sample(file_path, int(len(file_path) * train_rate))
    validation_flag = random.sample(list(set(file_path) - set(training_flag)), int(len(list(set(file_path) - set(training_flag))) * val_rate / (val_rate + test_rate)))
    # separate data into training, validation, test set based on patients
    for file_name in file_path:
        patient_id = file_name[-10:-4]
        seizure_indexs = generate_seizure_index(patient_id, df)
        eeg_data, ecg_data, res_data = read_data(file_name, eeg_signals, ecg_signals, res_signals)
        batch_x_eeg, batch_x_ecg, batch_x_res, batch_y = window_gen(eeg_data, ecg_data, res_data, seizure_indexs)
        if file_name in training_flag:
            training_set["y"] += batch_y
            training_set["x_eeg"] += batch_x_eeg
            training_set["x_ecg"] += batch_x_ecg
            training_set["x_res"] += batch_x_res
        elif file_name in validation_flag:
            validation_set["y"] += batch_y
            validation_set["x_eeg"] += batch_x_eeg
            validation_set["x_ecg"] += batch_x_ecg
            validation_set["x_res"] += batch_x_res
        else:
            test_set["y"] += batch_y
            test_set["x_eeg"] += batch_x_eeg
            test_set["x_ecg"] += batch_x_ecg
            test_set["x_res"] += batch_x_res
    return training_set, validation_set, test_set

training_set, validation_set, test_set = xy_gen(path, xlsx_path, sheet_name)

scio.savemat('./gen_dataset/training_set.mat', training_set)
scio.savemat('./gen_dataset/validation_set.mat', validation_set)
scio.savemat('./gen_dataset/test_set.mat', test_set)

print("data saved successfully")