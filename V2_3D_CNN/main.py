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

    model.add(Conv1D(128, 4, strides=2, activation='relu', padding='same'))
    model.add(Conv1D(128, 4, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(256, 2, strides=1, activation='relu', padding='same'))
    model.add(Conv1D(256, 2, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    return model

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 20 == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("./Model/each100Epochs/model_{}.hd5".format(epoch+1))

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
    saver = CustomSaver()
    #tensorboard = TensorBoard(log_dir="./logs")

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy', auc, f1])

    hist = model.fit(
        train_x,
        train_y,
        batch_size=2000,
        epochs=2000,
        validation_data=val_ds,
        #callbacks=[ckpt, tensorboard])
        callbacks=[ckpt, saver])

    #dict_keys(['val_loss', 'val_acc', 'val_auc', 'val_f1', 'loss', 'acc', 'auc', 'f1'])
    with open("./Model/epoch_information/epoch_information.txt", 'a') as f:
        for epo in range(len(hist.history['val_auc'])):
            s = "epochs: " + str(epo+1) + " loss: " + str(hist.history['loss'][epo])[0:6] + \
                " acc: " + str(hist.history['acc'][epo])[0:6] + " auc: " + str(hist.history['auc'][epo])[0:6] + \
                " f1: " + str(hist.history['f1'][epo])[0:6] + " val_loss: " + str(hist.history['val_loss'][epo])[0:6] + \
                " val_acc: " + str(hist.history['val_acc'][epo])[0:6] + " val_auc: " +\
                str(hist.history['val_auc'][epo])[0:6] + " val_f1: " + str(hist.history['val_f1'][epo])[0:6]
            f.write((s+'\n'))
    f.close()
