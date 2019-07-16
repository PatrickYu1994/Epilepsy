import keras
import scipy.io as scio
import keras.backend.tensorflow_backend as KFT
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.objectives import *
from keras.optimizers import SGD


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

#xs_val = np.append(np.append(validation_set['x_eeg'], validation_set['x_ecg'], axis=1), validation_set['x_res'], axis=1)
#ys_val = np.array(validation_set['y'])


# build model
model = Sequential()
# Dense layer to reduce dimensionality from 8500 to 4096
model.add(Dense(input_dim=xs.shape[1], output_dim=4096, activation='relu'))
# Reshape 2D to 3D
model.add(Reshape((64, 64), input_shape=(64*64,)))
# two 1024 conv
model.add(Conv1D(4096, 2, strides=2, activation='relu', padding='same'))
model.add(Conv1D(4096, 2, strides=2, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(2048, 4, strides=2, activation='relu', padding='same'))
model.add(Conv1D(2048, 4, strides=2, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(1024, 8, strides=2, activation='relu', padding='same'))
model.add(Conv1D(1024, 8, strides=2, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(512, 16, strides=2, activation='relu', padding='same'))
model.add(Conv1D(512, 16, strides=2, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 32, strides=2, activation='relu', padding='same'))
model.add(Conv1D(256, 32, strides=2, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.3))
model.add(Dense(1, activation='relu'))


optimizer = SGD(lr=1e0)
model.compile(loss='mean_squared_error',
              optimizer=optimizer)
history = model.fit(xs, ys, nb_epoch=50,
                    batch_size=xs.shape[0],
                    verbose=0)
model.save('./v1_2019_07_17')
print(history)

#tf.concat













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




