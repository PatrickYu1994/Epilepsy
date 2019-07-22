import numpy as np
import scipy.io as scio
from main2 import xs_gen

test_set = scio.loadmat("./gen_dataset/test_set.mat")

x, y = xs_gen(test_set)
print(x.shape) # (987, 500, 17)
#print(x.transpose(0,2,1).shape)
'''
print(x.shape[0])
print(x.shape[1])
print(x.shape[2])
print(y.shape)

y = np.array(test_set['y'])

a = np.array(test_set['x_eeg'])
b = np.array(test_set['x_ecg'])
c = np.array(test_set['x_res'])

x1 = np.concatenate((a, b, c), axis=2)
print(x1.shape)
print(y.shape)
'''