import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from PIL import Image
import tensorflow as tf 
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D, PReLU
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#                                   data
### ================================================================= ###
                    # P(pt, phi)
px1, py1 = np.genfromtxt('D:/Users/mathe/ML/EoS/DATA/EOSL/EOSL_samples_111.dat', usecols=(0, 1), unpack=True) # EOSL
px2, py2 = np.genfromtxt('D:/Users/mathe/ML/EoS/DATA/EOSQ/EOSQ_samples_111.dat', usecols=(0, 1), unpack=True) # EOSQ
print(px1[0], py1[0])
print(px1[1], py1[1])
print(px2[0], py2[0])
print(px2[1], py2[1])
print('-+-'*10)

def pt_calc(px, py):
    pt_ = [] # pt = sqrt(px² + py²)
    for i in range(len(px)):
        calc = np.sqrt((px[i])**2 + (py[i])**2)
        pt_ += [calc]
    return pt_
pt = pt_calc(px1, py1)
pt2 = pt_calc(px2, py2)
print(pt[0], pt[1])
print(pt2[0], pt2[1])
print('-+-'*10)

def phi_calc(px, py):
    phif_ = [] # phi = atan(py/px)
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
phif = phi_calc(px1, py1)
phif2 = phi_calc(px2, py2)
print(phif[0], phif[1])
print(phif2[0], phif2[1])

# first event from EOSL_111
pt_1 = np.linspace(pt[0], pt[1002])
phi_1 = np.linspace(phif[0], phif[1002])

                    # numpy.meshgrid
print(f'max: {max(pt_1)}, {max(phi_1)}')
print(f'min: {min(pt_1)}, {min(phi_1)}')

sns.scatterplot(data='D:/Users/mathe/ML/EoS/DATA/EOSL/EOSL_samples_111.dat', x=pt, y=phif, color='darkred')
plt.show()

                    ### Sample Control
# reading the data from "control.dat"
with open("D:/Users/mathe/ML/EoS/DATA/EOSL control/EOSL_samples_control_111.dat", "r") as control:
    events = [int(line.strip()) for line in control]
with open("D:/Users/mathe/ML/EoS/DATA/EOSQ control/EOSQ_samples_control_111.dat", "r") as control2:
    events2 = [int(line2.strip()) for line2 in control2]

# Create a list to fill with the data splited by the events
data_per_event = []
data_per_event2 = []
''
# scrool the event's list & split the data corresponding to each event
def event_append(data_per_event_, events, pt, phif):
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

# data_per_event is a list where each element is a tuple with the list's Pt & Phi of any event
# Getting acssess of Pt and phi values for any event and ploting:
event_1 = data_per_event[0]
pt_event_1 = event_1[0]
phi_event_1 = event_1[1]
print(len(pt_event_1))
print(len(phi_event_1))

event_2 = data_per_event[1]
pt_event_2 = event_2[0]
phi_event_2 = event_2[1]
print(len(pt_event_2))
print(len(phi_event_2))

event_100 = data_per_event[99]
pt_event_100 = event_100[0]
phi_event_100 = event_100[1]
print(len(pt_event_100))
print(len(phi_event_100))

plt.subplot(1, 1, 1)
sns.scatterplot(data='D:/Users/mathe/ML/EoS/DATA/EOSL/EOSL_samples_111.dat', x=pt_event_1, y=phi_event_1, color='green')
sns.scatterplot(data='D:/Users/mathe/ML/EoS/DATA/EOSL/EOSL_samples_111.dat', x=pt_event_2, y=phi_event_2, color='blue')
sns.scatterplot(data='D:/Users/mathe/ML/EoS/DATA/EOSL/EOSL_samples_111.dat', x=pt_event_100, y=phi_event_100, color='orange')
plt.show()
print('-=-'*20)

                    # Making the plots the way it should be
heatmap, xedges, yedges = np.histogram2d(pt_event_1, phi_event_1, bins=[40, 40])

fig, ax = plt.subplots(figsize=(10, 8))
im = plt.imshow(heatmap.T, origin='lower')

# Show all ticks and label them with the respective list entries
pt_form_num = [f'{numero:.2f}' for numero in pt_event_1]
pt_form_num.sort()
phi_form_num = [f'{numero:.2f}' for numero in phi_event_1]
phi_form_num.sort()
ax.set_xticks(np.arange(len(yedges)-1), labels=(pt_form_num[0:-10:25])) # 1034 // bin
ax.set_yticks(np.arange(len(xedges)-1), labels=(phi_form_num[0:-10:25]))
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

ax.set_title("Number of particles given transversial moment and azhimutal angle bin's for a event")
ax.set_xlabel("phi \n event 1")
ax.set_ylabel("Pt")
plt.tight_layout()
data_1 = plt.gcf() # SAVE FIG IN A VARIABLE SO I CAN SAVE IT LATER!
plt.show()

                    # Ultimos remendos do gráfico



                    # Criar input Data
def all_events(data_per_event_, nOevents=100):
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
print(len(pts[0]), len(phis[0]))
print(len(pts[1]), len(phis[1]))
print(len(pts2[0]), len(phis2[0]))
print(len(pts2[1]), len(phis2[1]))
print('\033[1;35m-=-\033[m'*20)

def create_data(pts_, phis_, nOevents=100, bin_=40):
    final_data = []
    for i in range(0, nOevents-1):

        heatmap, xedges, yedges = np.histogram2d(pts_[i], phis_[i], bins=[bin_, bin_])

        fig, ax = plt.subplots(figsize=(10, 8))
        im = plt.imshow(heatmap.T, origin='lower')

        pt_form_num = [f'{numero:.2f}' for numero in pt_event_1] ## trocar event_1 por todos!
        pt_form_num.sort()
        phi_form_num = [f'{numero:.2f}' for numero in phi_event_1]
        phi_form_num.sort()
        a = 1034 // bin_
        ax.set_xticks(np.arange(len(yedges)-1), labels=(pt_form_num[0:-10:a])) # 1034 / bin
        ax.set_yticks(np.arange(len(xedges)-1), labels=(phi_form_num[0:-10:a]))

        ax.set_title("Number of particles given transversial moment and azhimutal angle bin's for a event")
        ax.set_xlabel("phi \n event 1")
        ax.set_ylabel("Pt")
        plt.tight_layout()
#        data_n = plt.gcf() # SAVE FIG IN A VARIABLE SO I CAN SAVE IT LATER!
#        final_data += [data_n]
#        for i, figura in enumerate(final_data):
#            figura.savefig(f'figura_{i+1}.png')

        plt.title(f'Figura {i+1}')
        buffer = io.BytesIO() # save image in a memory buffer
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode('utf-8') # convert in base64

        final_data += [image_base64]
        plt.close()
    return final_data

data_eosl = create_data(pts, phis)
data_eosq = create_data(pts2, phis2)
print(len(data_eosl))
print(len(data_eosq))

"""for i in range(3):
    image_base64 = data_eosl[i]
    image_base642 = data_eosq[i]

    # base64 --> bin
    image_bin = base64.b64decode(image_base64)
    image_bin2 = base64.b64decode(image_base642)

    # bin --> Pilow
    image_pil = Image.open(io.BytesIO(image_bin))
    image_pil2 = Image.open(io.BytesIO(image_bin2))

    plt.imshow(image_pil)
    plt.axis('off')
    plt.show()
    plt.imshow(image_pil2)
    plt.axis('off')
    plt.show()"""

#                               network
### ================================================================= ###
# base64 --> bin
eosl_bin = [base64.b64decode(image_b) for image_b in data_eosl]
eosq_bin = [base64.b64decode(image_b2) for image_b2 in data_eosq]
print(len(eosl_bin))

# bin --> Pilow
eosl_pil = [Image.open(io.BytesIO(image_p)) for image_p in eosl_bin]
eosq_pil = [Image.open(io.BytesIO(image_p2)) for image_p2 in eosq_bin]
print(len(eosl_pil))

#preprocess
eosl_process = [imagem.resize((15, 48)) for imagem in eosl_pil]
eosq_process = [imagem2.resize((15, 48)) for imagem2 in eosq_pil]
print(len(eosl_process))

# PIL -->  Array Numpy
eosl_images = [np.array(imagem3) for imagem3 in eosl_process]
eosq_images = [np.array(imagem4) for imagem4 in eosq_process]
print(len(eosl_images))

# Labels
eosl_labels = [0] * len(eosl_images) # 0 --> EOSL /// 1 --> EOSQ
eosq_labels = [1] * len(eosq_images)

# split sets
X_train, X_val, Y_train, Y_val = train_test_split((eosl_images + eosq_images), (eosl_labels + eosq_labels), test_size=0.3, shuffle=False) # 0.3
# With n_samples=1, test_size=0.3 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.
print(f'\033[1;32mtrain shape:\033[m      {len(X_train)}')
print(f'\033[1;32mvalidation shape:\033[m {len(X_val)}')
X_train = np.array(X_train)
X_val = np.array(X_val)
Y_train = np.array(Y_train)
Y_val = np.array(Y_val)

            # MODEL archtecture
init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
model = tf.keras.Sequential([
    
    Conv2D(16, (8, 8), kernel_regularizer='l2', kernel_initializer=init, input_shape=(15, 48, 4)), # 15, 48, 1
    PReLU(),
    Dropout(0.2),
    BatchNormalization(),
    
    Conv2D(32, (7, 7), kernel_regularizer='l2', kernel_initializer=init),
    PReLU(),
    Dropout(0.2), 
    BatchNormalization(), 
    AveragePooling2D(), 
    
    Flatten(),
    Dropout(0.5), 
    BatchNormalization(),
    Dense(128, activation='relu'), # sigmoid
    
    Dense(1, activation='sigmoid')
    
])

print('\033[1;33m')
model.summary()
print('\033[m-=-' *20)


model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics='accuracy'
)

cb = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='auto')
Re = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, min_lr=1e-6)
batch_size = 4
h1 = model.fit(X_train, Y_train, epochs=500, validation_data=(X_val, Y_val), batch_size=batch_size, verbose=2, callbacks=[Re, cb])

plt.subplot(1, 2, 1)
plt.plot(h1.history['val_loss'], 'b', label='val')
plt.subplot(1, 2, 1)
plt.plot(h1.history['loss'], 'r', label='train')
plt.subplot(1, 2, 2)
plt.plot(h1.history['val_accuracy'], 'b', label='val')
plt.subplot(1, 2, 2)
plt.plot(h1.history['accuracy'], 'r', label='train')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

