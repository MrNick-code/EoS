import numpy as np
from os import walk
from os.path import join
from io import StringIO
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator as IDG
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D, PReLU, Conv3D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras import backend as K
from mpl_toolkits.mplot3d import Axes3D


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

'''X_train, Y_train = np.append(X_train1, X_train2), np.append(Y_train1, Y_train2)
X_val, Y_val = np.append(X_val1, X_val2), np.append(Y_val1, Y_val2)
X_test, Y_test = np.append(X_test1, X_test2), np.append(Y_test1, Y_test2)'''
X_train, Y_train = np.concatenate((X_train1, X_train2)), np.concatenate((Y_train1, Y_train2))
X_val, Y_val = np.concatenate((X_val1, X_val2)), np.concatenate((Y_val1, Y_val2))
X_test, Y_test = np.concatenate((X_test1, X_test2)), np.concatenate((Y_test1, Y_test2))
del X_train1, X_train2, Y_train1, Y_train2, X_test1, X_test2, Y_test1, Y_test2, X_val1, X_val2, Y_val1, Y_val2

X_train2, Y_train2 = np.concatenate((X_train1i, X_train2i)), np.concatenate((Y_train1i, Y_train2i))
X_val2, Y_val2 = np.concatenate((X_val1i, X_val2i)), np.concatenate((Y_val1i, Y_val2i))
X_test2, Y_test2 = np.concatenate((X_test1i, X_test2i)), np.concatenate((Y_test1i, Y_test2i))
del X_train1i, X_train2i, Y_train1i, Y_train2i, X_test1i, X_test2i, Y_test1i, Y_test2i, X_val1i, X_val2i, Y_val1i, Y_val2i

'''X_train, Y_train = np.concatenate((X_train, X_train2)), np.concatenate((Y_train, Y_train2))
X_val, Y_val = np.concatenate((X_val, X_val2)), np.concatenate((Y_val, Y_val2))
X_test, Y_test = np.concatenate((X_test, X_test2)), np.concatenate((Y_test, Y_test2))
del X_train2, Y_train2, X_val2, Y_val2, X_test2, Y_test2'''

'''X_train, Y_train = np.column_stack((X_train, X_train2)), np.column_stack((Y_train, Y_train2))
X_val, Y_val = np.column_stack((X_val, X_val2)), np.column_stack((Y_val, Y_val2))
X_test, Y_test = np.column_stack((X_test, X_test2)), np.column_stack((Y_test, Y_test2))
del X_train2, Y_train2, X_val2, Y_val2, X_test2, Y_test2'''

print(f'Train input shape:{np.shape(X_train)}')
print(f'Validation input shape: {np.shape(X_val)}')
print(f'Test input shape: {np.shape(X_test)}')
print(f'Train output shape: {np.shape(Y_train)}')
print(f'Validation output shape: {np.shape(Y_val)}')
print(f'Test output shape: {np.shape(Y_test)} \n')

init = tf.keras.initializers.RandomNormal(mean=.0, stddev=.01)
model_o = tf.keras.Sequential([  
    Conv2D(16, (8, 8), padding='same', kernel_regularizer='l2', kernel_initializer=init, input_shape=(60, 60, 4)), 
    Dropout(0.2),
    BatchNormalization(),
    PReLU(),
    
    Conv2D(32, (7, 7), padding='same', kernel_regularizer='l2', kernel_initializer=init), 
    Dropout(0.2), 
    BatchNormalization(), 
    AveragePooling2D(), 
    PReLU(),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5), 
    BatchNormalization(),
    
    Dense(3, activation='linear')   
])

print('\033[1;33m')
model_o.summary()
print('\033[m-=-' *20)

# p= 1: Manhattam (less outlier impact, less computational power required) | p=2: Euclidian (same escale. no correlation)
def minkowski_distance(y_true, y_pred, p=2):
    distances = []
    for i in range(3):
        distance = tf.reduce_mean(tf.abs(y_true[:, i] - y_pred[:, i]) ** p, axis=-1) ** (1/p)
        distances.append(distance)
    return tf.reduce_mean(distances) # means of the 3 distances (can be any other combination)

model_o.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=.0001),
    metrics=[minkowski_distance]
)

cb = EarlyStopping(monitor='val_loss', min_delta=.001, patience=50, mode='auto')
h1 = model_o.fit(x=X_train2, y=Y_train2, epochs=200, validation_data=(X_val2, Y_val2), verbose=2, callbacks=[cb], batch_size=64) #32 dps 64

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(h1.history['val_loss'], 'b', label='Validation Loss')
plt.plot(h1.history['loss'], 'r', label='Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(h1.history['val_minkowski_distance'], 'b', label='Validation Minkowski Distance')
plt.plot(h1.history['minkowski_distance'], 'r', label='Training Minkowski Distance')
plt.ylabel('Minkowski Distance')
plt.xlabel('Epochs')
plt.title('Training and Validation Minkowski Distance')
plt.legend()
plt.show()

predictions = model_o.predict(X_test2)

fig, axs = plt.subplots(1, 3, figsize=(14, 6))

for i in range(3):
    axs[i].scatter(Y_test2[:, i], predictions[:, i], c='blue', label=f'v_{i+2}²')
    axs[i].plot(Y_test2[:, i], Y_test2[:, i], color='red', linestyle='--') # ref line: what should be
    axs[i].set_xlabel(f'Real v_{i+2}²')
    axs[i].set_ylabel(f'Predicted v_{i+2}²')
    axs[i].legend()

plt.suptitle('Real vs. Predicted Values for Each Output Dimension')
plt.show()

# Movendo para tt pois percebi que preciso fazer certas alterações para a augmentation de nº of bin's ser usada.
