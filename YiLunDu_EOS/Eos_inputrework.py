import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import tensorflow as tf 
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D, PReLU, Conv3D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

if not __name__ == "__main__":
    dirname_test = "D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSL"
    for dirpath, dirnames, filenames in os.walk(dirname_test):
        print(dirpath)
        print(dirnames)
        print(filenames)
        print("-=-" * 20)
    print("-+-" * 20)
    for dirpath, dirnames, filenames in os.walk(dirname_test): 
        for names in filenames:
            print(os.path.join(dirpath, names))
    # the loop already considers all files.
    # but need to skip first.
    f = open(os.path.join("D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSL/Eosl_1/", 
                          'auau-music-1'), 'r')
    print(f'\033[1;35m{f.read()}')
    f.seek(0)
    print(f'\033[m{f.readlines()}')

    with open(os.path.join("D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSL/Eosl_1/", 
                          'auau-music-1'), 'r') as arquivo:
        conteudo = arquivo.read()
    if not conteudo:
        print("O arquivo está vazio.")
    else:
        linhas = conteudo.split('\n') # lembrete, importante!!!!!
        print("Número de linhas no arquivo:", len(linhas))
        for linha in linhas:
            print(linha)

class manipulation:

    def __init__(self, dirname):
        self.dirname = dirname
        self.get_columns()

    def get_columns(self):
        '''
            EOSL & EOSQ:
                20 Folders
                    100 events
                        ~4000 particles por event
        '''

        px, py, n_particles = [], [], []

        for dirpath, dirnames, filenames in os.walk(self.dirname):
            
            for names in filenames:

                full_path = os.path.join(dirpath, names)
                if '.dat' in names: # skip vn files
                    continue

                with open(full_path, 'r') as file_unit:
                    content = file_unit.read() # ready the files
                if not content:
                    print('the file is empty.')
                else:
                    lines = content.split('\n') # fix problems cuz \n
                    try:
                        header = int(lines[0].strip())
                    except ValueError:
                        print(f'error reading the header: {full_path}')


                lines_str = '\n'.join(lines)
                lines_float = [[float(num) for num in string.split()] for string in lines] # convert all values to float
                lines_str = '\n'.join([' '.join(map(str, line)) for line in lines_float]) # A função StringIO espera uma única string como entrada, não uma lista de listas de floats
                data = np.genfromtxt(io.StringIO(lines_str), usecols=(2, 3), skip_header=1, unpack=True)
                pxi, pyi = data[0], data[1]

                n_particles.append(header)
                px.append(pxi)
                py.append(pyi)

        return px, py, n_particles

    def pt_calc(self):
        '''
            Calculates transversial momentum
            pt = sqrt(px² + py²)
        '''
        pt_ = []
        px, py, n_particles = self.get_columns()

        for j in range(0, 2000): # 2000 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (e os outros)
            for i in range(len(px[j])):
                calc = np.sqrt((px[j][i])**2 + (py[j][i])**2)
                pt_ += [calc] 

        return pt_

    def phi_calc(self):
        '''
            Calculates azhimutal angle
            phi = atan(py/px)
        '''
        phif_ = []
        pi = np.pi
        px, py, n_particles = self.get_columns()

        for j in range(0, 2000):
            for i in range(len(px[j])):
                a = py[j][i]/px[j][i]

                if px[j][i] == 0:
                    if py[j][i] >= 0:
                        phi = pi/2
                    else:
                        phi = (pi/2)*3
                elif px[j][i] < 0:
                    if py[j][i] >= 0:
                        phi = np.arctan(a) + pi
                    else:
                        phi = np.arctan(a) + pi
                else:
                    if py[j][i] >= 0:
                        phi = np.arctan(a)
                    else:
                        phi = np.arctan(a) + 2*pi
                phif_ += [phi]

        return phif_

dir1 = "D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSL"
dir2 = "D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSQ"

eosl_data = manipulation(dir1)
pxl, pyl, particles_l = eosl_data.get_columns()
ptl = eosl_data.pt_calc()
phifl = eosl_data.phi_calc() # ~ 6.282 root

eosq_data = manipulation(dir2)
pxq, pyq, particles_q = eosq_data.get_columns()
ptq = eosq_data.pt_calc()
phifq = eosq_data.phi_calc()

print(len(ptl), len(phifl))
print(ptl[0], phifl[0])
print(particles_l)
print("-=-" * 20)

"""dir3 = "D:/Users/mathe/ML/EoS/DATA/EOSL_30_60centrality/EOSL_30-60centra"
eosl_data_3060 = manipulation(dir3)
px30, py30, particles_30 = eosl_data_3060.get_columns()
pt30 = eosl_data_3060.pt_calc()
phif30 = eosl_data_3060.phi_calc()
"""

### DATA IMPORT ###
data_per_event = []
data_per_event2 = []
#data_oer_event30 = []

class events_class:

    def __init__(self, data_per_event_, num_lines_per_event, pt, phif):
        self.data_per_event_ = data_per_event_
        self.num_lines_per_event = num_lines_per_event
        self.pt = pt
        self.phif = phif
        self.event_append()

    def event_append(self):
        '''
            Split the data corresponding to every event
        '''
        start = 0
        for num_lines in self.num_lines_per_event:
            Pt_event = self.pt[start:start + num_lines] 
            Phi_event = self.phif[start:start + num_lines]
            self.data_per_event_.append((Pt_event, Phi_event))
            start += num_lines
        return self.data_per_event_

    def all_events(self): 
        '''
            Create all events pt's and phi's
        '''
        data_per_event_ = self.event_append()
        event_n, pt_event_n, phi_event_n = [], [], []
        for i in range(0, 2000): #################################################### TESTE, ESTAVA 1 AQUI, TROCAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            event_i = data_per_event_[i]
            pt_event, phi_event = event_i[0], event_i[1]
            event_n += [event_i] # n --> i
            pt_event_n += [pt_event]
            phi_event_n += [phi_event]
        return event_n, pt_event_n, phi_event_n

eosl_event_data = events_class(data_per_event, particles_l, ptl, phifl)
images, pts, phis = eosl_event_data.all_events()

eosq_event_data = events_class(data_per_event2, particles_q, ptq, phifq)
images2, pts2, phis2 = eosq_event_data.all_events()

"""eos30_event_data = events_class(data_oer_event30, particles_30, pt30, phif30)
images3, pts3, phis3 = eos30_event_data.all_events()"""

# PLOT example
"""pt = pts[0]
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
"""

def create_data(pts_, phis_, nOevents=2000, bin_=(80, 80), save_path=None):
    final_data = []
    for i in range(0, nOevents): # there was a -1

        pt = pts_[i] # list index out of range
        phi = phis_[i]

        range_phi = (-np.pi/12, (2 * np.pi) + (np.pi/12))
        range_pt = (min(pt)-0.1, max(pt)+1)

        hist, xedges, yedges = np.histogram2d(phi, pt, bins=bin_, range=[range_phi, range_pt]) # object of type 'numpy.float64' has no len()
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

#        if i < 2:
#            plt.show()
#        else:
#            pass

#        data_n = plt.gcf() # SAVE FIG IN A VARIABLE SO I CAN SAVE IT LATER!
#        final_data += [data_n]
#        for i, figura in enumerate(final_data):
#            figura.savefig(f'figura_{i+1}.png')

        if save_path:
            filename = f'{save_path}/figure__{i+1}.png'
            plt.savefig(filename, format='png') # FileNotFoundError: [Errno 2] No such file or directory: 'D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_imgs80/figure__1.png'
            final_data.append(filename)
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            final_data.append(image_base64)
        plt.close()

    return final_data

#data_visualize_3060 = create_data(pts3, phis3, save_path="D:/Users/mathe/ML/EoS/DATA/EOSL_30_60centrality/savepath")

data_eosl = create_data(pts, phis, save_path="D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_imgs80")
data_eosq = create_data(pts2, phis2, save_path="D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_imgs80")







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
