'''
EOSX                        (X = L, Q)                  folder
    EOSX_N                  (N = 1, 2, 3, ..., 20)      sub-folder
        auau-music-M        (M = 1, 2, 3, ..., 100)     file
            L' lines        (L' ~ 4000)                 content
        EosX_Nvn                                        file
            M=100 lines                                 content

all files auau-music-M correspond to a single line of EosX_Nvn.
'''
EOSX = ["D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSL", "D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSQ"]
n = 20
m = 100

import numpy as np
from os import walk
from os.path import join
from io import StringIO

class get_values:
    '''
        by the end, create matrix with the data that we need:
            L' lines: particles
            3* columns: pt, phi, vn (all with same vn for each auau-music-M)
    '''

    def __init__(self, dirname):
        self.dirname = dirname
        self.get_initial_columns()

    def get_initial_columns(self):
        '''
        get inital columns to do calculations
        idéia: pegar as colunas como antes, multiplicar os vn pelo n_particles e atribuir a cada linha.

            EOSX:
                EOSX_N
                    auau-music-M
                        L' lines
        '''
        px, py, n_particles = [], [], []
        vn_expanded = []

        for dirpath, dirnames, filenames in walk(self.dirname):
            for names in filenames:
                full_path = join(dirpath, names)


                if '.dat' in names:
                    with open(full_path, 'r') as file_vn:
                        content_vn = file_vn.read()
                    print(f' vn path: {full_path}')
                    if not content_vn:
                        print('file_vn is empty.')
                    else:
                        lines_vn = content_vn.split('\n')
                        try:
                            headera = str(lines_vn[0].strip()) # not int. str because ievent
                            print(f'this is the header of an vn file: {headera}')
                        except:
                            print(f'error reading the vn header: {full_path}')

                    lines_str_vn = '\n'.join(lines_vn)
                    lines_f_vn = [[float(numa) for numa in stringa.split()] for stringa in lines_vn[1:]]
                    print(f'lines_f_vn first line is: {lines_f_vn[0]}')
                    lines_str_vn = '\n'.join([' '.join(map(str, linea)) for linea in lines_f_vn])
                    data_vn = np.genfromtxt(StringIO(lines_str_vn), usecols=(1, 2, 3), unpack=True) # no skip_header=1 because lines_vn[1:]
                    # considering the hypotesis that just need a single vn for each image, lets return the data_vn
                    '''v2, v3, v4 = data_vn[0], data_vn[1], data_vn[2]
                    print(f'len v2: {len(v2)}')
                    for i in range(0, len(v2)):
                        vn_expanded_temp = np.full(shape=(n_particles[i], 3), fill_value=[v2[i], v3[i], v4[i]])
                        vn_expanded.append(vn_expanded_temp)'''
                    file_vn.close()
                    continue


                with open(full_path, 'r') as file_unit:
                    content = file_unit.read()
                if not content:
                    print('file is empty.')
                else:
                    lines = content.split('\n')
                    try:
                        header = int(lines[0].strip())
                    except:
                        print(f'error reading the header: {full_path}')

                lines_str = '\n'.join(lines)
                lines_float = [[float(num) for num in string.split()] for string in lines] 
                lines_str = '\n'.join([' '.join(map(str, line)) for line in lines_float]) 
                data = np.genfromtxt(StringIO(lines_str), usecols=(2, 3), skip_header=1, unpack=True)
                pxi, pyi = data[0], data[1]

                n_particles.append(header)
                px.append(pxi)
                py.append(pyi)

                file_unit.close()

        return px, py, n_particles, vn_expanded, data_vn

    '''
    Talvez não seja necessário vn_expanded. Se eu for atrelar a função dele após criação das imagens, apenas 1 por imagem
    basta, ou seja, vn mesmo! No momento vn_expanded tras 1 por linha. Indiferente dessa decisão, talvez eu possa trazer essa
    informação junto do gráfico, de alguma forma

    Por causa disso, estou retornando data_vn também, posso decidir mais tarde'''

    def pt_calc(self):
        pt_ = []
        px, py, n_particles, vn_expanded, data_vn = self.get_initial_columns()
        for j in range(n*m):
            for i in range(len(px[j])):
                calc = np.sqrt((px[j][i])**2 + (py[j][i])**2)
                pt_ += [calc] 
        return pt_

    def phi_calc(self):
        phif_ = []
        pi = np.pi
        px, py, n_particles, vn_expanded, data_vn = self.get_initial_columns()
        for j in range(n*m):
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

eosl_data = get_values(EOSX[0])
pxl, pyl, particles_l, vn_expanded_l, vn_l = eosl_data.get_initial_columns()
ptl = eosl_data.pt_calc()
phifl = eosl_data.phi_calc()

'''eosq_data = get_values(EOSX[1])
pxq, pyq, particles_q, vn_expanded_q, vn_q = eosq_data.get_initial_columns()
ptq = eosq_data.pt_calc()
phifq = eosq_data.phi_calc()'''

print(f'len ptl: {len(ptl)} \n len phifl: {len(phifl)} \n len vn_l: {np.shape(vn_l)} \n') # 8215158 8215158 (3, 100)
print(f'ptl0: {ptl[0]} \n phifl0: {phifl[0]}') # 0.422869 3.6069899
print(f'particles_l: {np.shape(particles_l)} \n vn_l: {vn_l}') # (2000, ) all
print("-=-" * 20)

'''
So, I have all the images and the labels now. But how can I relate every image to it's label, since its an image X list?
'''
'''
Try to take the saved images (not splited) and attech to vn_x
'''

