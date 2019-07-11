import os
import numpy as np
import pandas as pd
from mne.io import read_raw_edf

# Hyperparameters
path = "../../.."
#data_path = "../../../CPM027.edf"
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
p_n_rate = 1/2 # seizure:non-seizure rate = 1:2
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
    batch_x_eeg = []
    batch_x_ecg = []
    batch_x_res = []
    batch_y = []
    for index in range(0, (len(eeg_data) - window_size), stride):
        y = 0 # where 0 stands no_seizure, 1 stands for seizure
        for s_arr in seizure_indexs:
            if (index > s_arr[0]) and ((index + window_size) < s_arr[1]):
                y = 1
        batch_y.append([y])
        batch_x_eeg.append(eeg_data[index: (index + window_size)].values.
                           reshape(1, window_size*len(eeg_data.columns))[0])
        batch_x_ecg.append(ecg_data[index: (index + window_size)].values.
                           reshape(1, window_size*len(ecg_data.columns))[0])
        batch_x_res.append(res_data[index: (index + window_size)].values.
                           reshape(1, window_size * len(res_data.columns))[0])
    return batch_x_eeg, batch_x_ecg, batch_x_res, batch_y

def xy_gen(path, xlsx_path, sheet_name = "Seizure Information"):
    df = read_csv(xlsx_path, sheet_name)
    for file_name in read_file_name(path):
        patient_id = file_name[-10:-4]
        seizure_indexs = generate_seizure_index(patient_id, df)
        eeg_data, ecg_data, res_data = read_data(file_name, eeg_signals, ecg_signals, res_signals)
        batch_x_eeg, batch_x_ecg, batch_x_res, batch_y = \
            window_gen(eeg_data, ecg_data, res_data, seizure_indexs)
        print(len(batch_y))
        break

xy_gen(path, xlsx_path, sheet_name)
#print(np.random.randn(2,3))



