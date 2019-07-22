import os
import numpy as np
import scipy.io as scio
import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
import keras.backend.tensorflow_backend as KFT
from sklearn.metrics import roc_auc_score
from keras.callbacks import TensorBoard
import datetime
from utils import auc, f1
from time import time

'''
CONV1D
channel 1: xxxx
channel 2: xxxx
channel 3: xxxx
channel 4: xxxx
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KFT.set_session(sess)

# load preprocessing data
training_set = scio.loadmat("./gen_dataset/training_set.mat")
validation_set = scio.loadmat("./gen_dataset/validation_set.mat")
print("loading successfully")

# generate corresponding x and y
def xs_gen(data_set):
    eeg = np.array(data_set['x_eeg'])
    ecg = np.array(data_set['x_ecg'])
    res = np.array(data_set['x_res'])
    xs = np.concatenate((eeg, ecg, res), axis=2) # (no rows, signals size, no channels) (e.g. 987, 500, 17)
    #ys = keras.utils.np_utils.to_categorical(training_set['y'], 2)
    ys = np.array(data_set['y'])
    return xs, ys

def build_model(xs):
    # build model
    model = Sequential()

    model.add(Conv1D(16, 16, strides=2, activation='relu', input_shape=(xs.shape[1], xs.shape[2])))
    model.add(Conv1D(16, 16, strides=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 8, strides=2, activation='relu', padding='same'))
    model.add(Conv1D(64, 8, strides=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, 4, strides=1, activation='relu', padding='same'))
    model.add(Conv1D(128, 4, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(256, 2, strides=1, activation='relu', padding='same'))
    model.add(Conv1D(256, 2, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    return model

if __name__ == '__main__':
    train_x, train_y = xs_gen(training_set)
    val_ds = xs_gen(validation_set)
    model = build_model(train_x)
    print(model.summary())
    date = datetime.date.today().strftime("%Y%m%d")
    filepath = "./Model/best_model/" + date + 'best_model.{epoch:02d}-{val_auc:.4f}.h5'
    ckpt = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                           monitor='val_auc',
                                           save_best_only=True,
                                           verbose=1,
                                           mode='max')

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    #adam = keras.optimizers.adam(lr=1e-5)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', auc, f1])

    history = model.fit(
        train_x,
        train_y,
        batch_size=2000,
        epochs=1500,
        validation_data=val_ds,
        callbacks=[ckpt, tensorboard])

    model.save('./Model/v1_2019_07_17_pnrate_1_2')


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




