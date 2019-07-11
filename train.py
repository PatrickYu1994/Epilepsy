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
def read_data(path, eeg_signals, ecg_signals, res_signals):
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
            #print(type(pd.to_timedelta(df["Recording Start"].astype(str))))
            temp[row["Patient ID"]] = row["Recording Start"]
    for _, row in df.iterrows():
        if pd.isnull(row["Recording Start"]) and not pd.isnull(row["Patient ID"]):
            row["Recording Start"] = temp[row["Patient ID"]]
    return df
