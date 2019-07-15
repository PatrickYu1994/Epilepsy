import scipy.io as scio
import tensorflow as tf

training_set = scio.loadmat("./gen_dataset/training_set.mat")
validation_set = scio.loadmat("./gen_dataset/validation_set.mat")
test_set = scio.loadmat("./gen_dataset/test_set.mat")

print(len(training_set['x_eeg']))
print(len(validation_set['x_eeg']))
print(len(test_set['x_eeg']))

