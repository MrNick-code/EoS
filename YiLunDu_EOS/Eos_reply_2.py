import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from PIL import Image
import tensorflow as tf 
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D, PReLU, Conv3D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

### COLUMS MANIPULATION ###
px1, py1 = np.genfromtxt('D:/Users/mathe/ML/EoS/DATA/EOSL/EOSL_samples_111.dat', usecols=(0, 1), unpack=True) # EOSL
px2, py2 = np.genfromtxt('D:/Users/mathe/ML/EoS/DATA/EOSQ/EOSQ_samples_111.dat', usecols=(0, 1), unpack=True) # EOSQ

class calculus:

    def __init__(self, px, py):
        self.px = px
        self.py = py

    def pt_calc(self, px, py):
        '''
            Calculates transversial momentum
            pt = sqrt(px² + py²)
        '''
        pt_ = []
        for i in range(len(px)):
            calc = np.sqrt((px[i])**2 + (py[i])**2)
            pt_ += [calc]
        return pt_

    def phi_calc(self, px, py):
        '''
            Calculates azhimutal angle
            phi = atan(py/px)
        '''
        phif_ = []
        pi = np.pi
        for i in range(len(px)):
            if px[i] == 0:
                if py[i] >= 0:
                    phi = pi/2
                else:
                    phi = (pi/2)*3
            elif px[i] < 0:
                if py[i] >= 0:
                    phi = np.arctan(py[i]/px[i]) + pi
                else:
                    phi = np.arctan(py[i]/px[i]) + pi
            else:
                if py[i] >= 0:
                    phi = np.arctan(py[i]/px[i])
                else:
                    phi = np.arctan(py[i]/px[i]) + 2*pi
            phif_ += [phi]
        return phif_

eosl_data = calculus(px1, py1)
pt = eosl_data.pt_calc(px1, py1)
phif = eosl_data.phi_calc(px1, py1)

eosq_data = calculus(px2, py2)
pt2 = eosq_data.pt_calc(px2, py2)
phif2 = eosq_data.phi_calc(px2, py2)

### EVENTS MANIPULATION ###
with open("D:/Users/mathe/ML/EoS/DATA/EOSL control/EOSL_samples_control_111.dat", "r") as control:
    events = [int(line.strip()) for line in control]
with open("D:/Users/mathe/ML/EoS/DATA/EOSQ control/EOSQ_samples_control_111.dat", "r") as control2:
    events2 = [int(line2.strip()) for line2 in control2]

data_per_event = []
data_per_event2 = []

def event_append(data_per_event_, events, pt, phif):
    '''
        Split the data corresponding to every event
    '''
    start = 0
    for event in events:
        end = start + event
        Pt_event = pt[start:end]
        Phi_event = phif[start:end]
        data_per_event_.append((Pt_event, Phi_event))
        start = end
    return data_per_event_

data_per_event = event_append(data_per_event, events, pt, phif)
data_per_event2 = event_append(data_per_event2, events2, pt2, phif2)

def all_events(data_per_event_, nOevents=100):
    '''
        Create all events pt's and phi's
    '''
    event_n, pt_event_n, phi_event_n = [], [], []
    for i in range(0, nOevents-1):
        event_i = data_per_event_[i]
        pt_event, phi_event = event_i[0], event_i[1]
        event_n += [event_i] # n --> i
        pt_event_n += [pt_event]
        phi_event_n += [phi_event]
    return event_n, pt_event_n, phi_event_n

images, pts, phis = all_events(data_per_event)
images2, pts2, phis2 = all_events(data_per_event2)

# PLOT example
pt = pts[0]
phi = phis[0]

range_phi = (-np.pi/12, (2 * np.pi) + (np.pi/12))
range_pt = (min(pt)-0.1, max(pt)+1)

hist, xedges, yedges = np.histogram2d(phi, pt, bins=(40, 40), range=[range_phi, range_pt])
plt.imshow(hist.T, extent=[range_phi[0], range_phi[1], range_pt[0], range_pt[1]], cmap='viridis', aspect='auto', origin='lower')
plt.xlim(*range_phi)
plt.ylim(*range_pt)
x_labels = [range_phi[0] + (range_phi[1] - range_phi[0]) / 3, range_phi[0] + 2 * (range_phi[1] - range_phi[0]) / 3]
y_labels = [range_pt[0] + (range_pt[1] - range_pt[0]) / 3, range_pt[0] + 2 * (range_pt[1] - range_pt[0]) / 3]

plt.xticks(x_labels)
plt.yticks(y_labels)
plt.title('Nº of particles in an event')
plt.xlabel('Azimuthal Angle')
plt.ylabel('Traversial Momentum')
plt.show()

### IMAGE MANIPULATION ###
def create_data(pts_, phis_, nOevents=100, bin_=(40, 40)):
    final_data = []
    for i in range(0, nOevents-1):

        pt = pts_[i]
        phi = phis_[i]

        range_phi = (-np.pi/12, (2 * np.pi) + (np.pi/12))
        range_pt = (min(pt)-0.1, max(pt)+1)

        hist, xedges, yedges = np.histogram2d(phi, pt, bins=bin_, range=[range_phi, range_pt])
        plt.imshow(hist.T, extent=[range_phi[0], range_phi[1], range_pt[0], range_pt[1]], cmap='viridis', aspect='auto', origin='lower')
        plt.xlim(*range_phi)
        plt.ylim(*range_pt)
        x_labels = [range_phi[0] + (range_phi[1] - range_phi[0]) / 3, range_phi[0] + 2 * (range_phi[1] - range_phi[0]) / 3]
        y_labels = [range_pt[0] + (range_pt[1] - range_pt[0]) / 3, range_pt[0] + 2 * (range_pt[1] - range_pt[0]) / 3]

        plt.xticks(x_labels)
        plt.yticks(y_labels)
        plt.title(f'Nº of particles in an event ({i+1})')
        plt.xlabel('Azimuthal Angle')
        plt.ylabel('Traversial Momentum')

        if i == 0:
            plt.show()
        else:
            pass

#        data_n = plt.gcf() # SAVE FIG IN A VARIABLE SO I CAN SAVE IT LATER!
#        final_data += [data_n]
#        for i, figura in enumerate(final_data):
#            figura.savefig(f'figura_{i+1}.png')

        buffer = io.BytesIO() # save image in a memory buffer
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode('utf-8') # convert in base64

        final_data += [image_base64]
        plt.close()
    return final_data

data_eosl = create_data(pts, phis)
data_eosq = create_data(pts2, phis2)

### DATA SET ###
# base64 --> bin
eosl_bin = [base64.b64decode(image_b) for image_b in data_eosl]
eosq_bin = [base64.b64decode(image_b2) for image_b2 in data_eosq]

# bin --> Pilow
eosl_pil = [Image.open(io.BytesIO(image_p)) for image_p in eosl_bin]
eosq_pil = [Image.open(io.BytesIO(image_p2)) for image_p2 in eosq_bin]

#preprocess
eosl_process = [imagem.resize((48, 15)) for imagem in eosl_pil]
eosq_process = [imagem2.resize((48, 15)) for imagem2 in eosq_pil]

# PIL -->  Array Numpy
eosl_images = [np.array(imagem3) for imagem3 in eosl_process]
eosq_images = [np.array(imagem4) for imagem4 in eosq_process]

# Labels
eosl_labels = [0] * len(eosl_images) # 0 --> EOSL /// 1 --> EOSQ
eosq_labels = [1] * len(eosq_images)

# split sets
X = eosl_images + eosq_images
Y = eosl_labels + eosq_labels
Y = to_categorical(Y, num_classes=2) 
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, shuffle=True) 

print(f'\033[1;32mtrain shape:\033[m      {len(X_train)}')
print(f'\033[1;32mvalidation shape:\033[m {len(X_val)}')


### Convolutional Neural Network ###
X_train = np.array(X_train)
X_val = np.array(X_val)
Y_train = np.array(Y_train)
Y_val = np.array(Y_val)
print(f'Array 4D input shape (X_train): {X_train.shape}') # (138, 48, 15, 4)

            # MODEL archtecture
init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
model = tf.keras.Sequential([
    
    Conv2D(16, (8, 8), padding='same', kernel_regularizer='l2', kernel_initializer=init, input_shape=(15, 48, 4)), 
    Dropout(0.2),
    BatchNormalization(),
    PReLU(),
    
    Conv2D(32, (7, 7), padding='same', kernel_regularizer='l2', kernel_initializer=init),
    Dropout(0.2), 
    BatchNormalization(), 
    AveragePooling2D(), 
    PReLU(),
    
    Flatten(),
    Dense(128, activation='sigmoid'),
    BatchNormalization(),
    Dropout(0.5), 
    
    Dense(2, activation='linear') 
    
])

print('\033[1;33m')
model.summary()
print('\033[m-=-' *20)


model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),
    metrics='accuracy'
)

cb = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='auto')
batch_size = 1
h1 = model.fit(X_train, Y_train, epochs=500, validation_data=(X_val, Y_val), batch_size=batch_size, verbose=2, callbacks=[cb])

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
