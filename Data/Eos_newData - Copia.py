import os
import numpy as np
import matplotlib.pyplot as plt

### DATA EXPORT ###
class manipulation:

    def __init__(self, dirname, ext):
        self.dirname = dirname
        self.ext = ext
        self.get_columns()

    def get_columns(self):
        '''
        For every folder inside EOSL/EOSQ data
        For files without "Eos" ste, It get the columns (auau-music-n) and header (nº particles)
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
                    lines = file_unit.readlines()

                header = lines[0].strip()
                data = np.genfromtxt(lines[1:], usecols=(2, 3), unpack=True)
                pxi, pyi = data[0], data[1]
                n_particles.append(header)
                px.append(pxi)
                py.append(pyi)

        #print(len(px[0]))
        #print(len(px))

        return np.array(px), np.array(py), n_particles
    
    def pt_calc(self):
        '''
            Calculates transversial momentum
            pt = sqrt(px² + py²)
        '''
        pt_Sub, pt_ = [], []
        px, py, n_particles = self.get_columns()
        for j in range(0, 20):
            for i in range(len(px[j])):
                calc = np.sqrt((px[j][i])**2 + (py[j][i])**2)
                pt_Sub += [calc]

            pt_ += [pt_Sub]

        return pt_
    
    def phi_calc(self):
        '''
            Calculates azhimutal angle
            phi = atan(py/px)
        '''
        phif_Sub, phif_ = [], []
        pi = np.pi
        px, py, n_particles = self.get_columns()
        for j in range(0, 20):
            for i in range(len(px[j])):
                if px[j][i] == 0: # The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
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
                phif_Sub += [phi]

            phif_ += [phif_Sub]

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
particles_l = [int(value) for value in particles_l]
ptl = eosl_data.pt_calc()
phifl = eosl_data.phi_calc() # ~ 6.282 root

eosq_data = manipulation(dir2, '.dat')
pxq, pyq, particles_q = eosq_data.get_columns()
particles_q = [int(value) for value in particles_q]
ptq = eosq_data.pt_calc()
phifq = eosq_data.phi_calc()

print(len(ptl), len(phifl))
print(ptl[0], phifl[0])
print(particles_l[0]) # Now I have pt, phi and nº particles.

### DATA IMPORT ###
data_per_event = []
data_per_event2 = []

def event_append(data_per_event_, num_lines_per_event, pt, phif):
    '''
        Split the data corresponding to every event
    '''
    start = 0
    for num_lines in num_lines_per_event:
        Pt_event = pt[start:start + num_lines] # unsupported operand type(s) for +: 'int' and 'str'
        Phi_event = phif[start:start + num_lines]
        data_per_event_.append((Pt_event, Phi_event))
        start += num_lines
    return data_per_event_

data_per_event = event_append(data_per_event, particles_l, ptl, phifl)
data_per_event2 = event_append(data_per_event2, particles_q, ptq, phifq)

def all_events(data_per_event_, nOevents=2000): 
    '''
        Create all events pt's and phi's
    '''
    event_n, pt_event_n, phi_event_n = [], [], []
    for i in range(0, nOevents): ################## TINHA UM -1 AQUI (novents-1) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        event_i = data_per_event_[i]
        pt_event, phi_event = event_i[0], event_i[1]
        event_n += [event_i] # n --> i
        pt_event_n += [pt_event]
        phi_event_n += [phi_event]
    return event_n, pt_event_n, phi_event_n

images, pts, phis = all_events(data_per_event)
images2, pts2, phis2 = all_events(data_per_event2)

# PLOT example
pt = pts[0][0]
phi = phis[0][0] ######################################### ERA SÓ 1 [0] AQUI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

range_phi = (-np.pi/12, (2 * np.pi) + (np.pi/12))
range_pt = (min(pt)-0.1, max(pt)+1) # TypeError: unsupported operand type(s) for -: 'list' and 'float'

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







pt = pts[0][1]
phi = phis[0][1] ######################################### ERA SÓ 1 [0] AQUI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

range_phi = (-np.pi/12, (2 * np.pi) + (np.pi/12))
range_pt = (min(pt)-0.1, max(pt)+1) # TypeError: unsupported operand type(s) for -: 'list' and 'float'

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

pt = pts[0][2]
phi = phis[0][2] ######################################### ERA SÓ 1 [0] AQUI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

range_phi = (-np.pi/12, (2 * np.pi) + (np.pi/12))
range_pt = (min(pt)-0.1, max(pt)+1) # TypeError: unsupported operand type(s) for -: 'list' and 'float'

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

pt = pts[15][0] # List index out of range
phi = phis[15][0] ######################################### ERA SÓ 1 [0] AQUI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

range_phi = (-np.pi/12, (2 * np.pi) + (np.pi/12))
range_pt = (min(pt)-0.1, max(pt)+1) # TypeError: unsupported operand type(s) for -: 'list' and 'float'

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
