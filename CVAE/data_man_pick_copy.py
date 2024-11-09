# Taken from: flow_idea_tt.py # copy of data_man_pick.py
import numpy as np
from os import walk
from os.path import join
from io import StringIO
import os
from PIL import Image
from sklearn.model_selection import train_test_split


EOSX = ["C:\\Users\\mathe\\faculdade\\ML\\EoS\\DATA\\EOS_vn\\EOSL", "C:\\Users\\mathe\\faculdade\\ML\\EoS\\DATA\\EOS_vn\\EOSQ"]
'''EOSXimgs_40 = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_40b", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_low_40"]
EOSXimgs_50 = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_50", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_low_50"]
# paths = EOSXimgs_40 e EOSXimgs_50
# shape = (50, 50, 4) e (60, 60, 4)'''

class get_values(object): # o (object) é coisa de python 2

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

def get_data(paths, shape):
    
    X_L, Y_L, vn_L = get_values(EOSX[0]).append_imageLabel(paths[0], shape)
    del vn_L
    X_Q, Y_Q, vn_Q = get_values(EOSX[1]).append_imageLabel(paths[1], shape)
    del vn_Q

    # reshape

    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_L, Y_L, test_size=100, random_state=42)
    X_train1, X_val1, Y_train1, Y_val1 = train_test_split(X_train1, Y_train1, test_size=0.3, random_state=42)
    del X_L, Y_L
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_Q, Y_Q, test_size=100, random_state=42)
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train2, Y_train2, test_size=0.3, random_state=42)
    del X_Q, Y_Q

    X_train = np.concatenate((X_train1, X_train2))
    X_val = np.concatenate((X_val1, X_val2))
    X_test = np.concatenate((X_test1, X_test2))
    Y_train = np.concatenate((Y_train1, Y_train2))
    Y_val = np.concatenate((Y_val1, Y_val2))
    Y_test = np.concatenate((Y_test1, Y_test2))

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
