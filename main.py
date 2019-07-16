import keras
import scipy.io as scio
import keras.backend.tensorflow_backend as KFT
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.objectives import *


# load preprocessing data
training_set = scio.loadmat("./gen_dataset/training_set.mat")
validation_set = scio.loadmat("./gen_dataset/validation_set.mat")
test_set = scio.loadmat("./gen_dataset/test_set.mat")
print("loading successfully")

# merge all signal data together (training set & validation set)
xs = np.append(np.append(training_set['x_eeg'], training_set['x_ecg'], axis=1), training_set['x_res'], axis=1)
# add additional dimension [36174, 8500, 1], input shape is [8500, 1]
#xs = np.expand_dims(xs, axis=2)
#input_shape = [xs.shape[1], xs.shape[2]]

ys = np.array(training_set['y'])
xs_val = np.append(np.append(validation_set['x_eeg'], validation_set['x_ecg'], axis=1), validation_set['x_res'], axis=1)

ys_val = np.array(validation_set['y'])




# build model
model = Sequential()
model.add(Dense(input_dim=xs.shape[1], output_dim=4096, activation='relu'))
model.add(Conv1D())













#print(xs.shape[0])
#print(xs.shape)
#print(ys.shape)



#dt = tf.data.Dataset.from_tensor_slices((test_set['x_ecg'], test_set['x_eeg'], test_set['x_res'], test_set['y']))

#def build_model():

#print(test_set.keys())
#print(test_set['x_eeg'])

#a = [[1,2,3], [3,2,1]]
#b = [[1,2,3], [3,2,1]]
#c = [[1,2,3], [3,2,1]]
#print(np.append(np.append(a,b,axis=1), c, axis=1))




