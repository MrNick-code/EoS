from tensorboard.plugins.hparams import api as hp
from keras.preprocessing.image import ImageDataGenerator as IDG
import tensorflow as tf 
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D, PReLU
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
import pickle
import numpy as np

train_dg, val_dg = IDG(horizontal_flip=True, rescale=1/255, vertical_flip=True), IDG(horizontal_flip=True, rescale=1/255, vertical_flip=True)
test_dg = IDG(horizontal_flip=True, rescale=1/255, vertical_flip=True)

train_set = train_dg.flow_from_directory("D:/Users/mathe/ML/EoS/IMG_DATA/DATA_CROP/train40",
                                    target_size=(60, 60), batch_size=64, class_mode="categorical", color_mode="rgb", shuffle=True, seed=42) # 15 48
val_set = val_dg.flow_from_directory("D:/Users/mathe/ML/EoS/IMG_DATA/DATA_CROP/val40",
                                    target_size=(60, 60), batch_size=64, class_mode="categorical", color_mode="rgb", shuffle=True, seed=42) # bs 64
test_set = test_dg.flow_from_directory("D:/Users/mathe/ML/EoS/IMG_DATA/DATA_CROP/test_test",
                                    target_size=(60, 60), batch_size=64, class_mode="categorical", color_mode="rgb", shuffle=True, seed=42) # 15 48


if __name__ == "__main__":
    print(f'\033[1;32mtrain shape:\033[m      {len(train_set)}') # 2794 1206
    print(f'\033[1;32mvalidation shape:\033[m {len(val_set)}')

HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.3, 0.7))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adamax']))

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric('accuracy', display_name='Acc')],
  )

def train_test_model(hparams):
  
    init = tf.keras.initializers.RandomNormal(mean=.0, stddev=.01)
    model = tf.keras.models.Sequential([
    
        Conv2D(16, (8, 8), padding='same', kernel_regularizer='l2', kernel_initializer=init, input_shape=(60, 60, 3)), 
        Dropout(hparams[HP_DROPOUT]),
        BatchNormalization(),
        PReLU(),
    
        Conv2D(32, (7, 7), padding='same', kernel_regularizer='l2', kernel_initializer=init), 
        Dropout(hparams[HP_DROPOUT]), 
        BatchNormalization(), 
        AveragePooling2D(), 
        PReLU(),
    
        Flatten(),
        Dense(128, activation='relu'), # sigmoid
        Dropout(hparams[HP_DROPOUT]), 
        BatchNormalization(),
    
        Dense(2, activation='linear') 
    ])

    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    cb = EarlyStopping(monitor='val_loss', min_delta=.01, patience=50, mode='auto')
    model.fit(train_set, epochs=5, validation_data=(val_set), verbose=2, callbacks=[cb])
    _, accuracy = model.evaluate(val_set) # should be test set

    return accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar('accuracy', accuracy, step=10)

'''model.fit(
    ...,
    callbacks=[
        tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
    ],
)'''

session_num = 0

for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
            HP_DROPOUT: dropout_rate,
            HP_OPTIMIZER: optimizer,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning/' + run_name, hparams)
        session_num += 1

# TensorBoard Analysis with the VScode extension:
# $ Launch TensorBoard
# $ tensorboard --logdir logs/hparam_tuning
# $ tensorboard --logdir logs/fit (???)
# $ tensorboard --logdir logs/gradient_tape
