from keras.preprocessing.image import ImageDataGenerator as IDG
import tensorflow as tf 
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D, PReLU
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
import pickle
import numpy as np

train_dg, val_dg = IDG(horizontal_flip=True, rescale=1/255, vertical_flip=True), IDG(horizontal_flip=True, rescale=1/255, vertical_flip=True)

train_set = train_dg.flow_from_directory("D:/Users/mathe/ML/EoS/IMG_DATA/DATA_CROP/train",
                                    target_size=(60, 60), batch_size=64, class_mode="categorical", color_mode="rgb", shuffle=True, seed=42) # 15 48
val_set = val_dg.flow_from_directory("D:/Users/mathe/ML/EoS/IMG_DATA/DATA_CROP/val",
                                    target_size=(60, 60), batch_size=64, class_mode="categorical", color_mode="rgb", shuffle=True, seed=42) # bs 64

if __name__ == "__main__":
    print(f'\033[1;32mtrain shape:\033[m      {len(train_set)}') # 2794 1206
    print(f'\033[1;32mvalidation shape:\033[m {len(val_set)}')

            # MODEL archtecture
def create_model():
    init = tf.keras.initializers.RandomNormal(mean=.0, stddev=.01)
    model_o = tf.keras.Sequential([
        
        Conv2D(16, (8, 8), padding='same', kernel_regularizer='l2', kernel_initializer=init, input_shape=(60, 60, 3)), 
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

    '''print('\033[1;33m')
    model_o.summary()
    print('\033[m-=-' *20)'''

    model_o.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adamax(learning_rate=.0001), # ADAMAX 0.0001
        metrics='accuracy'
    )
    return model_o

model_o = KerasClassifier(build_fn=create_model)

### Grid Search Application
'''try:
    serialized_obj = pickle.dumps(train_set)
    print("O objeto é serializável!")
except pickle.PickleError as e:
    print("Ocorreu um erro ao serializar o objeto:", e)'''

'''e_seris_train = []
e_seris_val = []

for item in train_set:
    e_seri = pickle.dumps(item) # ação excede memória
    e_seris_train.append(e_seri)
for item in val_set:
    e_seri2 = pickle.dumps(item)
    e_seris_val.append(e_seri2)'''

X = np.array(train_set)  # Se suas imagens já estiverem no formato aceito pelo modelo 
# Unable to allocate 2.64 MiB for an array with shape (64, 60, 60, 3) and data type float32 !!!!!!!!!!!!!!!!!!!!!!!!!!!
Y = np.concatenate([np.zeros(1397), np.ones(1397)])  # Cria um array de 2000 zeros seguido por 2000 uns

bs = [16, 32, 64, 128]
param_grid = {'batch_size': bs}

grid = GridSearchCV(estimator=model_o, param_grid=param_grid, n_jobs=-1, cv=3)
# cb = EarlyStopping(monitor='val_loss', min_delta=.01, patience=50, mode='auto') # patience=50 for 300 epo
grid_result = grid.fit(X, Y, epochs=50, validation_data=None, verbose=2)

# Grid Search Ressults
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
