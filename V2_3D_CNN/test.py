import os
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
import mne
import random
import scipy.io as scio

data_path = "../../../CPM002.edf"
eeg_signals = ['C3', 'F7', 'F4', 'C4', 'Fz', 'Cz', 'Pz', 'Fp1',
               'P3', 'Fp2', 'P4', 'F3', 'F8']
ecg_signals = ['ECG']
res_signals = ['THO-', 'THO+', 'Air Flow']

def read_data(data_path, eeg_signals, ecg_signals, res_signals):
    rawData = read_raw_edf(data_path)
    tmp = rawData.to_data_frame()
    eeg_data = pd.DataFrame(mne.filter.filter_data(np.array(tmp[eeg_signals].T), 250, l_freq=40, h_freq=1).T)
    #eeg_data = tmp[eeg_signals]
    ecg_data = tmp[ecg_signals]
    res_data = tmp[res_signals]
    return eeg_data, ecg_data, res_data
eeg, ecg, res = read_data(data_path, eeg_signals, ecg_signals, res_signals)

print(eeg)
print(eeg.shape)
print(type(eeg))
