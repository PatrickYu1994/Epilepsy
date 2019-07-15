import scipy.io as scio
import tensorflow as tf
import numpy as np

#training_set = scio.loadmat("./gen_dataset/training_set.mat")
#validation_set = scio.loadmat("./gen_dataset/validation_set.mat")
#test_set = scio.loadmat("./gen_dataset/test_set.mat")

#dt = tf.data.Dataset.from_tensor_slices((test_set['x_ecg'], test_set['x_eeg'], test_set['x_res'], test_set['y']))

#def build_model():


a = [[1,2,3], [3,4,5]]
print(np.array(a))
