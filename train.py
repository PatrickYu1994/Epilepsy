import os
import numpy as np
import mne
from mne.io import read_raw_edf

path = "../../.."
data_path = "../../../CPM027.edf"

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
def read_data(path):
    eeg_signals = ['C3', 'F7', 'F4', 'C4', 'Fz', 'Cz', 'Pz',
                   'Fp1', 'P3', 'Fp2', 'P4', 'F3', 'F8']
    ecg_signals = ['ECG']
    res_signals = ['THO-', 'THO+', 'Air Flow']

    rawData = read_raw_edf(data_path)
    tmp = rawData.to_data_frame()
    eeg_data = tmp[eeg_signals]
    ecg_data = tmp[ecg_signals]
    res_data = tmp[res_signals]

    return eeg_data, ecg_data, res_data

#def generate_windows

print(read_file_name(path))

