from keras.preprocessing.image import ImageDataGenerator as IDG
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D, PReLU, Conv3D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

train_dg, val_dg = IDG(horizontal_flip=True, rescale=1/255, vertical_flip=True), IDG(horizontal_flip=True, rescale=1/255, vertical_flip=True)
''' original!!
train_set = train_dg.flow_from_directory("D:/Users/mathe/ML/EoS/IMG_DATA/DATA_CROP/train80",
                                    target_size=(60, 60), batch_size=64, class_mode="categorical", color_mode="rgb", shuffle=True, seed=42) # 15 48
val_set = val_dg.flow_from_directory("D:/Users/mathe/ML/EoS/IMG_DATA/DATA_CROP/val80",
                                    target_size=(60, 60), batch_size=64, class_mode="categorical", color_mode="rgb", shuffle=True, seed=42) # bs 64
'''
# teste 17/05/2024
train_set = train_dg.flow_from_directory("D:\\Users\\mathe\\ML\\EoS\\IMG_DATA\\teste_delete\\train_",
                                    target_size=(50, 50), batch_size=64, class_mode="categorical", color_mode="grayscale", shuffle=True, seed=42) # 15 48
val_set = val_dg.flow_from_directory("D:\\Users\\mathe\\ML\\EoS\\IMG_DATA\\teste_delete\\val_",
                                    target_size=(50, 50), batch_size=64, class_mode="categorical", color_mode="grayscale", shuffle=True, seed=42)

if __name__ == "__main__":
    print(f'\033[1;32mtrain shape:\033[m      {len(train_set)}') # 2794 1206
    print(f'\033[1;32mvalidation shape:\033[m {len(val_set)}')

            # MODEL archtecture
init = tf.keras.initializers.RandomNormal(mean=.0, stddev=.01)
model_o = tf.keras.Sequential([
    
    Conv2D(16, (8, 8), padding='same', kernel_regularizer='l2', kernel_initializer=init, input_shape=(50, 50, 1)), 
    Dropout(0.2),
    BatchNormalization(),
    PReLU(),
    
    Conv2D(32, (7, 7), padding='same', kernel_regularizer='l2', kernel_initializer=init), 
    Dropout(0.2), 
    BatchNormalization(), 
    AveragePooling2D(), 
    PReLU(),
    
    Flatten(),
    Dense(128, activation='relu'), # sigmoid
    Dropout(0.5), 
    BatchNormalization(),
    
    Dense(2, activation='linear') 
    
])

model_ = tf.keras.Sequential([

    Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(60, 60, 3)),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    #Conv2D(512, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), padding='same'),
    Flatten(),
    #Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),

    Dense(2, activation='linear')

])

print('\033[1;33m')
model_o.summary()
print('\033[m-=-' *20)

model_o.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=.0001), # ADAMAX 0.0001
    metrics='accuracy'
)

cb = EarlyStopping(monitor='val_loss', min_delta=.01, patience=50, mode='auto')
h1 = model_o.fit(train_set, epochs=40, validation_data=(val_set), verbose=2, callbacks=[cb])

plt.subplot(1, 2, 1)
plt.plot(h1.history['val_loss'], 'b', label='val')
plt.subplot(1, 2, 1)
plt.plot(h1.history['loss'], 'r', label='train')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.subplot(1, 2, 2)
plt.plot(h1.history['val_accuracy'], 'b', label='val')
plt.subplot(1, 2, 2)
plt.plot(h1.history['accuracy'], 'r', label='train')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend()
plt.show()
