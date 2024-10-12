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
#tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
#session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

'''session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.per_process_gpu_memory_fraction = 0.3
config = tf.estimator.RunConfig(session_config=session_config)'''

EOSX = ["D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSL", "D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSQ"]
EOSXimgs_40 = ["D:/Users/mathe/ML/EoS/IMG_DATA/DATA_CROP/train40/EOSL", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_imgs"]
EOSXimgs_50 = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_imgs50", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_imgs50"]
EOSXimgs_80 = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_imgs80", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_imgs80"]

EOSx_low = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_40", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_low_40"]

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
                        headera = str(lines_vn[0].strip()) # not int. str because ievent
                    except:
                        print(f'error reading the vn header: {full_path}')

                lines_str_vn = '\n'.join(lines_vn)
                lines_f_vn = [[float(numa) for numa in stringa.split()] for stringa in lines_vn[1:]]
                print(f'lines_f_vn first line: {lines_f_vn[0]}')
                lines_str_vn = '\n'.join([' '.join(map(str, linea)) for linea in lines_f_vn])
                data_vn_temp = np.genfromtxt(StringIO(lines_str_vn), usecols=(1, 2, 3), unpack=True) # no skip_header=1 because lines_vn[1:]
            
                data_vn.append(data_vn_temp)

                file_vn.close()
        
        data_vn = np.concatenate(data_vn, axis=1)

        return data_vn

    def append_imageLabel(self, imgs_path):
        X = []
        labels = []
        imgs = os.listdir(imgs_path)
        vn = self.get_initial_columns()
        print(f'vn* shape: {np.shape(vn)} \n') # 3, 2000

        for filename in imgs:
            if filename.endswith(".png"): 
                img = Image.open(os.path.join(imgs_path, filename))
                #img = img.resize((width, height)) 
                img_array = np.array(img) 
                X.append(img_array)

                index = imgs.index(filename)
                label = vn[:, index]

                labels.append(label)

        X = np.array(X, dtype=float) # ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2000,) + inhomogeneous part.
        labels = np.array(labels) ### parece que ta misturando float com string no X

        return X, labels, vn

#with tf.device('/job:localhost/replica:0/task:0/device:GPU:1'):
#X40l, labels40l, vn40l = get_values(EOSX[0]).append_imageLabel(EOSXimgs_40[0])
#X40q, labels40q, vn40q = get_values(EOSX[1]).append_imageLabel(EOSXimgs_40[1])
#X40 = np.append(X40l, X40q, axis=0)
#labels40 = np.append(labels40l, labels40q, axis=0)

X40lowL, labels40lowL, vn40l = get_values(EOSX[0]).append_imageLabel(EOSx_low[0])

print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n')
#with tf.device('/job:localhost/replica:0/task:0/device:GPU:1'):
X_train, X_test, y_train, y_test = train_test_split(X40lowL, labels40lowL, test_size=0.2, random_state=42)
del X40lowL
del labels40lowL
del vn40l

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))

#X_train = np.array(X_train).astype('float32')
#X_test = np.array(X_test).astype('float32')

#with tf.device('/job:localhost/replica:0/task:0/device:GPU:1'):
# X_train = np.array(X_train).reshape(-1, a*b*c)
'''train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
valid_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
del X_train
del y_train
del X_test
del y_test'''


############################################################

init = tf.keras.initializers.RandomNormal(mean=.0, stddev=.01)
# OOM when allocating tensor with shape[2457600,128] and type float on /job:localhost/replica:0/task:0/device:CPU:0 by allocator cpu [Op:AddV2]
# "The most expedient way is probably to reduce the batch size. It'll run slower, but use less memory."
model_o = tf.keras.Sequential([
    
    Conv2D(16, (8, 8), padding='same', kernel_regularizer='l2', kernel_initializer=init, input_shape=(50, 50, 4)), 
    Dropout(0.2),
    BatchNormalization(),
    PReLU(),
    
    Conv2D(32, (7, 7), padding='same', kernel_regularizer='l2', kernel_initializer=init), 
    Dropout(0.2), 
    BatchNormalization(), 
    AveragePooling2D(), 
    PReLU(),
    
    Flatten(),
    Dense(64, activation='relu'), # sigmoid
    Dropout(0.5), 
    BatchNormalization(),
    
    Dense(3, activation='linear') 
    
])

print('\033[1;33m')
model_o.summary()
print('\033[m-=-' *20)

model_o.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=.0001), # ADAMAX 0.0001
    metrics='mean_absolute_error'
)

cb = EarlyStopping(monitor='val_loss', min_delta=.001, patience=50, mode='auto')
#with tf.device('/job:localhost/replica:0/task:0/device:GPU:1'):
h1 = model_o.fit(x=X_train, y=y_train, epochs=150, validation_data=(X_test, y_test), verbose=2, callbacks=[cb], batch_size=32)
# OOM when allocating tensor with shape[2457600,128] and type float on /job:localhost/replica:0/task:0/device:CPU:0 by allocator cpu
#	 [[{{node gradient_tape/sequential/dense/MatMul/MatMul_1}}]]

plt.subplot(1, 2, 1)
plt.plot(h1.history['val_loss'], 'b', label='val')
plt.subplot(1, 2, 1)
plt.plot(h1.history['loss'], 'r', label='train')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.subplot(1, 2, 2)
plt.plot(h1.history['val_mean_absolute_error'], 'b', label='val')
plt.subplot(1, 2, 2)
plt.plot(h1.history['mean_absolute_error'], 'r', label='train')
plt.ylabel('MAE')
plt.xlabel('epochs')
plt.legend()
plt.show()

# ValueError: Input 0 of layer "sequential" is incompatible with the layer: 
#  expected shape=(None, 48, 64, 4), found shape=(32, 480, 640, 4)
