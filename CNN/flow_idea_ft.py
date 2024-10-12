import numpy as np
from os import walk
from os.path import join
from io import StringIO
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D, PReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, Adamax, RMSprop, SGD
from keras import regularizers as rgl

EOSX = ["D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSL", "D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSQ"]
EOSXimgs_40 = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_40b", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_low_40"]
EOSXimgs_50 = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_50", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_low_50"]

class get_values(object):

    def __init__(self, dirname):
        self.dirname = dirname

    def get_initial_columns(self):

        data_vn = []

        for dirpath, dirnames, filenames in walk(self.dirname):
            for names in filenames:
                full_path = join(dirpath, names)

                if '.dat' not in names:
                    continue

                with open(full_path, 'r') as file_vn:
                    content_vn = file_vn.read()
                print(f' vn path: {full_path}')
                if not content_vn:
                    print('file_vn is empty.')
                else:
                    lines_vn = content_vn.split('\n')
                    try:
                        headera = str(lines_vn[0].strip())
                    except:
                        print(f'error reading the vn header: {full_path}')

                lines_str_vn = '\n'.join(lines_vn)
                lines_f_vn = [[float(numa) for numa in stringa.split()] for stringa in lines_vn[1:]]
                print(f'lines_f_vn first line: {lines_f_vn[0]}')
                lines_str_vn = '\n'.join([' '.join(map(str, linea)) for linea in lines_f_vn])
                data_vn_temp = np.genfromtxt(StringIO(lines_str_vn), usecols=(1, 2, 3), unpack=True)
            
                data_vn.append(data_vn_temp)

                file_vn.close()
        
        data_vn = np.concatenate(data_vn, axis=1)

        return data_vn

    def append_imageLabel(self, imgs_path, compatibility=(50, 50, 4)):
        X = []
        labels = []
        imgs = os.listdir(imgs_path)
        vn = self.get_initial_columns()
        print(f'vn* shape: {np.shape(vn)} \n')

        for filename in imgs:
            if filename.endswith(".png"): 
                img = Image.open(os.path.join(imgs_path, filename))
                img_array = np.array(img) 

                if img_array.shape == compatibility:  # compatibility
                    X.append(img_array)

                    index = imgs.index(filename)
                    label = vn[:, index]

                    labels.append(label)
                else:
                    print(f"Ignoring {filename} cuz of incompatible shape: {img_array.shape}")

        X = np.array(X, dtype=float)
        labels = np.array(labels)

        return X, labels, vn

X_40lowL, Y_40lowL, vn_40lowL = get_values(EOSX[0]).append_imageLabel(EOSXimgs_40[0], (50, 50, 4))
del vn_40lowL
X_40lowQ, Y_40lowQ, vn_40lowQ = get_values(EOSX[1]).append_imageLabel(EOSXimgs_40[1], (50, 50, 4))
del vn_40lowQ

X_50lowL, Y_50lowL, vn_50lowL = get_values(EOSX[0]).append_imageLabel(EOSXimgs_50[0], (60, 60, 4))
del vn_50lowL
X_50lowQ, Y_50lowQ, vn_50lowQ = get_values(EOSX[1]).append_imageLabel(EOSXimgs_50[1], (60, 60, 4))
del vn_50lowQ

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_40lowL, Y_40lowL, test_size=100, random_state=42)
X_train1, X_val1, Y_train1, Y_val1 = train_test_split(X_train1, Y_train1, test_size=0.3, random_state=42)
del X_40lowL, Y_40lowL
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_40lowQ, Y_40lowQ, test_size=100, random_state=42)
X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train2, Y_train2, test_size=0.3, random_state=42)
del X_40lowQ, Y_40lowQ

X_train1i, X_test1i, Y_train1i, Y_test1i = train_test_split(X_50lowL, Y_50lowL, test_size=100, random_state=42)
X_train1i, X_val1i, Y_train1i, Y_val1i = train_test_split(X_train1i, Y_train1i, test_size=0.3, random_state=42)
del X_50lowL, Y_50lowL
X_train2i, X_test2i, Y_train2i, Y_test2i = train_test_split(X_50lowQ, Y_50lowQ, test_size=100, random_state=42)
X_train2i, X_val2i, Y_train2i, Y_val2i = train_test_split(X_train2i, Y_train2i, test_size=0.3, random_state=42)
del X_50lowQ, Y_50lowQ

print(np.shape(X_test1), np.shape(X_test2))

X_train1 = np.concatenate((X_train1, X_train2))
X_val1 = np.concatenate((X_val1, X_val2))
X_test1 = np.concatenate((X_test1, X_test2))
Y_train1 = np.concatenate((Y_train1, Y_train2))
Y_val1 = np.concatenate((Y_val1, Y_val2))
Y_test1 = np.concatenate((Y_test1, Y_test2))

X_train2 = np.concatenate((X_train1i, X_train2i))
X_val2 = np.concatenate((X_val1i, X_val2i))
X_test2 = np.concatenate((X_test1i, X_test2i))
Y_train2 = np.concatenate((Y_train1i, Y_train2i))
Y_val2 = np.concatenate((Y_val1i, Y_val2i))
Y_test2 = np.concatenate((Y_test1i, Y_test2i))

print(np.shape(X_test1), np.shape(X_test2))

# p= 1: Manhattam (less outlier impact, less computational power required) | p=2: Euclidian (same escale. no correlation)
def minkowski_distance(y_true, y_pred, p=2):
    distances = []
    for i in range(3):
        distance = tf.reduce_mean(tf.abs(y_true[:, i] - y_pred[:, i]) ** p, axis=-1) ** (1/p)
        distances.append(distance)
    return tf.reduce_mean(distances) # means of the 3 distances (can be any other combination)

def create_model(optimizer='adam', dropout_rate=0.2):
    input1 = tf.keras.layers.Input(shape=(50, 50, 4))
    #input2 = tf.keras.layers.Input(shape=(60, 60, 4))

    # Camadas convolucionais para processar o primeiro conjunto de imagens
    conv1 = tf.keras.layers.Conv2D(16, (8, 8), padding='same', kernel_regularizer='l2')(input1)
    dropout1 = tf.keras.layers.Dropout(dropout_rate)(conv1)
    batch_norm1 = tf.keras.layers.BatchNormalization()(dropout1)
    prelu1 = tf.keras.layers.PReLU()(batch_norm1)

    conv2 = tf.keras.layers.Conv2D(32, (7, 7), padding='same', kernel_regularizer='l2')(prelu1)
    dropout2 = tf.keras.layers.Dropout(dropout_rate)(conv2)
    batch_norm2 = tf.keras.layers.BatchNormalization()(dropout2)
    average_pooling = tf.keras.layers.AveragePooling2D()(batch_norm2)
    prelu2 = tf.keras.layers.PReLU()(average_pooling)

    flatten1 = tf.keras.layers.Flatten()(prelu2)

    # Camadas convolucionais para processar o segundo conjunto de imagens
    '''conv3 = tf.keras.layers.Conv2D(16, (8, 8), padding='same', kernel_regularizer='l2')(input2)
    dropout3 = tf.keras.layers.Dropout(dropout_rate)(conv3)
    batch_norm3 = tf.keras.layers.BatchNormalization()(dropout3)
    prelu3 = tf.keras.layers.PReLU()(batch_norm3)

    conv4 = tf.keras.layers.Conv2D(32, (7, 7), padding='same', kernel_regularizer='l2')(prelu3)
    dropout4 = tf.keras.layers.Dropout(dropout_rate)(conv4)
    batch_norm4 = tf.keras.layers.BatchNormalization()(dropout4)
    average_pooling2 = tf.keras.layers.AveragePooling2D()(batch_norm4)
    prelu4 = tf.keras.layers.PReLU()(average_pooling2)

    flatten2 = tf.keras.layers.Flatten()(prelu4)'''

    # Concatenar as saídas das camadas convolucionais
    #concatenated = tf.keras.layers.concatenate([flatten1, flatten2])

    # Camadas densas finais
    dense1 = tf.keras.layers.Dense(64, activation='relu')(flatten1) # concatenated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dropout5 = tf.keras.layers.Dropout(dropout_rate)(dense1)
    batch_norm5 = tf.keras.layers.BatchNormalization()(dropout5)

    output = tf.keras.layers.Dense(3, activation='linear')(batch_norm5)

    # Criar modelo
    model = tf.keras.Model(inputs=input1, outputs=output) # [1, 2] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    # Compilar modelo
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=optimizer,
        metrics=[minkowski_distance]
    )

    return model

def new_create_model(optimizer, dropout_rate, activation_function, learning_rate, regularizer, k_size, filters):
    input1 = tf.keras.layers.Input(shape=(50, 50, 4))

    conv1 = tf.keras.layers.Conv2D(filters, k_size, padding='same', kernel_regularizer=rgl.l2(regularizer), activation=activation_function)(input1)
    dropout1 = tf.keras.layers.Dropout(dropout_rate)(conv1)
    batch_norm1 = tf.keras.layers.BatchNormalization()(dropout1)
    #prelu1 = tf.keras.layers.PReLU()(batch_norm1)

    conv2 = tf.keras.layers.Conv2D(filters, k_size, padding='same', kernel_regularizer=rgl.l2(regularizer), activation=activation_function)(batch_norm1)
    dropout2 = tf.keras.layers.Dropout(dropout_rate)(conv2)
    batch_norm2 = tf.keras.layers.BatchNormalization()(dropout2)
    average_pooling = tf.keras.layers.AveragePooling2D()(batch_norm2)
    #prelu2 = tf.keras.layers.PReLU()(average_pooling)

    flatten1 = tf.keras.layers.Flatten()(average_pooling)

    dense1 = tf.keras.layers.Dense(64, activation=activation_function)(flatten1)
    dropout5 = tf.keras.layers.Dropout(dropout_rate)(dense1)
    batch_norm5 = tf.keras.layers.BatchNormalization()(dropout5)

    output = tf.keras.layers.Dense(3, activation='linear')(batch_norm5)

    model = tf.keras.Model(inputs=input1, outputs=output) 

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'adamax':
        opt = Adamax(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    else:
        raise ValueError('Invalid optimizer')

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=opt,
        metrics=['acc']
    )

    return model

# keras wrapper to use GridSearchCV
model = KerasClassifier(build_fn=new_create_model, verbose=0)

param_grid = {
    'optimizer': ['adam', 'adamax'],
    'dropout_rate': [0.2, 0.5, 0.8]
}

new_param_grid = {
    'optimizer': ['adam', 'adamax'],
    'dropout_rate': [0.2],
    'activation_function': ['relu', 'elu', 'selu'],
    'learning_rate': [.01, .001],
    'regularizer': [.01, .0001],
    'k_size': [(4, 4), (7, 7)],
    'filters': [16, 32]
}
'''new_param_grid = {
    'optimizer': ['adam', 'adamax', 'rmsprop', 'sgd'],
    'dropout_rate': [0.2, 0.5, 0.8],
    'activation_function': ['relu', 'linear', 'tanh', 'elu', 'selu'],
    'learning_rate': [.1, .01, .001],
    'regularizer': [.01, .001, .0001],
    'k_size': [(2, 2), (3, 3), (4, 4), (7, 7)],
    'filters': [8, 16, 32, 64]
}'''
# 'architecture': ['VGG', 'ResNet', 'Inception'] 

grid = GridSearchCV(estimator=model, param_grid=new_param_grid, cv=2, error_score='raise')

grid_result = grid.fit(X_train1, Y_train1, validation_data=(X_val1, Y_val1), epochs=100)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

'''
input1 = tf.keras.layers.Input(shape=(50, 50, 4))
input2 = tf.keras.layers.Input(shape=(60, 60, 4))

conv1 = tf.keras.layers.Conv2D(16, (8, 8), padding='same', kernel_regularizer='l2')(input1)
dropout1 = tf.keras.layers.Dropout(0.2)(conv1)
batch_norm1 = tf.keras.layers.BatchNormalization()(dropout1)
prelu1 = tf.keras.layers.PReLU()(batch_norm1)

conv2 = tf.keras.layers.Conv2D(32, (7, 7), padding='same', kernel_regularizer='l2')(prelu1)
dropout2 = tf.keras.layers.Dropout(0.2)(conv2)
batch_norm2 = tf.keras.layers.BatchNormalization()(dropout2)
average_pooling = tf.keras.layers.AveragePooling2D()(batch_norm2)
prelu2 = tf.keras.layers.PReLU()(average_pooling)

flatten1 = tf.keras.layers.Flatten()(prelu2)

conv3 = tf.keras.layers.Conv2D(16, (8, 8), padding='same', kernel_regularizer='l2')(input2)
dropout3 = tf.keras.layers.Dropout(0.2)(conv3)
batch_norm3 = tf.keras.layers.BatchNormalization()(dropout3)
prelu3 = tf.keras.layers.PReLU()(batch_norm3)

conv4 = tf.keras.layers.Conv2D(32, (7, 7), padding='same', kernel_regularizer='l2')(prelu3)
dropout4 = tf.keras.layers.Dropout(0.2)(conv4)
batch_norm4 = tf.keras.layers.BatchNormalization()(dropout4)
average_pooling2 = tf.keras.layers.AveragePooling2D()(batch_norm4)
prelu4 = tf.keras.layers.PReLU()(average_pooling2)

flatten2 = tf.keras.layers.Flatten()(prelu4)

concatenated = tf.keras.layers.concatenate([flatten1, flatten2])

dense1 = tf.keras.layers.Dense(64, activation='relu')(concatenated)
dropout5 = tf.keras.layers.Dropout(0.5)(dense1)
batch_norm5 = tf.keras.layers.BatchNormalization()(dropout5)

output = tf.keras.layers.Dense(3, activation='linear')(batch_norm5)

# p= 1: Manhattam (less outlier impact, less computational power required) | p=2: Euclidian (same escale. no correlation)
def minkowski_distance(y_true, y_pred, p=2):
    distances = []
    for i in range(3):
        distance = tf.reduce_mean(tf.abs(y_true[:, i] - y_pred[:, i]) ** p, axis=-1) ** (1/p)
        distances.append(distance)
    return tf.reduce_mean(distances) # means of the 3 distances (can be any other combination)

def create_model():
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adamax(learning_rate=.0001),
        metrics=[minkowski_distance]
    )
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)

param_grid = {
    'batch_size': [32, 64],
    'epochs': [50, 100],
    'p': [1, 2] 
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit([X_train1, X_train2], [Y_train1, Y_train2])
# Cannot have number of splits n_splits=3 greater than the number of samples: n_samples=2.

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# TensorBoard Analysis with the VScode extension:
# $ Launch TensorBoard
# $ tensorboard --logdir logs/hparam_tuning
# $ tensorboard --logdir logs/fit (???)
# $ tensorboard --logdir logs/gradient_tape

'''
