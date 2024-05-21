import os
import numpy as np
from io import StringIO

# extend [1, 2, 3, 4, 5, 6] //// append [1, 2, 3, [4, 5, 6]]
class manipulation:

    def __init__(self, dirname, ext):
        self.dirname = dirname
        self.ext = ext
        self.get_columns()

    def get_columns(self):
        '''
        For every folder inside EOSL/EOSQ data
        For files without "ext" extension, It get the columns (auau-music-n)
        Pxja and Pykb (a/b represents the folder and j/k the event)
        '''
        px = []
        py = []
        n_particles = []

        for path, dirc, files in os.walk(self.dirname):

            for name in files:
                #if name.endswith(self.ext):
                #    continue
                if 'Eos' in name:
                    continue

                full_path = os.path.join(path, name)
                with open(full_path, 'r') as file_unit:
                    content = file_unit.read()

                # Use StringIO to process pretending it is a file
                file_content, file_content2 = StringIO(content), StringIO(content)
                header = open(file_content2, 'r').readline().strip()
                data = np.genfromtxt(file_content, usecols=(2, 3), skip_header=1, unpack=True)
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
        for j in range(0, 20):
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
        for j in range(0, 20):
            for i in range(len(px[j])):
                if px[j][i] == 0:
                    if py[j][i] >= 0:
                        phi = pi/2
                    else:
                        phi = (pi/2)*3
                elif px[j][i] < 0:
                    if py[j][i] >= 0:
                        phi = np.arctan(py[j][i]/px[j][i]) + pi
                    else:
                        phi = np.arctan(py[j][i]/px[j][i]) + pi
                else:
                    if py[j][i] >= 0:
                        phi = np.arctan(py[j][i]/px[j][i])
                    else:
                        phi = np.arctan(py[j][i]/px[j][i]) + 2*pi
                phif_ += [phi]
        return phif_

dir1 = "D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSL"
dir2 = "D:/Users/mathe/ML/EoS/DATA/EOS_vn/EOSQ"

#event1 = manipulation(dir1, '.dat')
#px_data, py_data = event1.get_columns()
#print(len(px_data))
#print(px_data)
#print(px_data[0])
#print(len(px_data[0]))

eosl_data = manipulation(dir1, '.dat')
pxl, pyl, particles_l = eosl_data.get_columns()
ptl = eosl_data.pt_calc()
phifl = eosl_data.phi_calc() # ~ 6.282 root

eosq_data = manipulation(dir2, '.dat')
pxq, pyq, particles_q = eosq_data.get_columns()
ptq = eosq_data.pt_calc()
phifq = eosq_data.phi_calc()

print(len(ptl), len(phifl))
print(ptl[0], phifl[0])
print(particles_l)
