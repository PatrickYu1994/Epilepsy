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

def xy_gen(path, xlsx_path, sheet_name = "Seizure Information"):
    df = read_csv(xlsx_path, sheet_name)
    for file_name in read_file_name(path):
        patient_id = file_name[-10:-4]
        seizure_indexs = generate_seizure_index(patient_id, df)
        eeg_data, ecg_data, res_data = read_data(file_name, eeg_signals, ecg_signals, res_signals)
        print(type(eeg_data[0:1]))

xy_gen(path, xlsx_path, sheet_name)

