import os
import scipy.io as scio
import tensorflow as tf
from keras.models import *
from keras.layers import *
import keras.backend.tensorflow_backend as KFT

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KFT.set_session(sess)

# load preprocessing data
training_set = scio.loadmat("./gen_dataset/training_set.mat")
validation_set = scio.loadmat("./gen_dataset/validation_set.mat")
#test_set = scio.loadmat("./gen_dataset/test_set.mat")
print("loading successfully")

def xs_gen(training_set):
    # merge all signal data together (training set & validation set)
    xs = np.append(np.append(training_set['x_eeg'], training_set['x_ecg'], axis=1), training_set['x_res'], axis=1)
    # add additional dimension [36174, 8500, 1], input shape is [8500, 1]
    #xs = np.expand_dims(xs, axis=2)
    #input_shape = [xs.shape[1], xs.shape[2]]
    ys = np.array(training_set['y'])
    return xs, ys

def xs_val_gen(validation_set):
    xs_val = np.append(np.append(validation_set['x_eeg'], validation_set['x_ecg'], axis=1), validation_set['x_res'], axis=1)
    ys_val = np.array(validation_set['y'])
    return xs_val, ys_val

def build_model(xs):
    # build model
    model = Sequential()
    # Dense layer to reduce dimensionality from 8500 to 4096
    #model.add(Dense(input_dim=xs.shape[1], output_dim=4096, activation='relu'))
    #model.add(Dense(input_dim=xs.shape[1], output_dim=1024, activation='relu'))
    model.add(Reshape((xs.shape[1], 1), input_shape=(xs.shape[1],)))
    # Reshape 2D to 3D
    #model.add(Reshape((4096, 1), input_shape=(64*64,)))
    #model.add(Reshape((1024, 1), input_shape=(32*32,)))
    # two 1024 conv
    #model.add(Conv1D(4096, 32, strides=2, activation='relu', padding='same'))
    #model.add(Conv1D(4096, 32, strides=2, activation='relu', padding='same'))
    #model.add(MaxPooling1D(2))
    #model.add(Conv1D(2048, 16, strides=2, activation='relu', padding='same'))
    #model.add(Conv1D(2048, 16, strides=2, activation='relu', padding='same'))
    #model.add(MaxPooling1D(2))
    #model.add(Conv1D(1024, 8, strides=2, activation='relu', padding='same'))
    #model.add(Conv1D(1024, 8, strides=2, activation='relu', padding='same'))
    #model.add(MaxPooling1D(2))
    model.add(Conv1D(1024, 16, strides=2, activation='relu', padding='same'))
    model.add(Conv1D(1024, 16, strides=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    #model.add(Conv1D(512, 4, strides=2, activation='relu', padding='same'))
    #model.add(Conv1D(512, 4, strides=2, activation='relu', padding='same'))
    #model.add(MaxPooling1D(2))
    model.add(Conv1D(512, 8, strides=2, activation='relu', padding='same'))
    model.add(Conv1D(512, 8, strides=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    ######################################################################
    model.add(Conv1D(256, 4, strides=2, activation='relu', padding='same'))
    model.add(Conv1D(256, 4, strides=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 2, strides=1, activation='relu', padding='same'))
    model.add(Conv1D(128, 2, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    ######################################################################
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

if __name__ == '__main__':
    train_ds = xs_gen(training_set)
    val_ds = xs_val_gen(validation_set)

    model = build_model(train_ds[0])

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        train_ds[0],
        train_ds[1],
        batch_size=500,
        epochs=500,
        validation_data=val_ds,
        verbose=1)
    model.save('./Model/v1_2019_07_17')
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




