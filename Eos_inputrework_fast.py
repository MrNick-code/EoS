import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

class manipulation:

    def __init__(self, dirname):
        self.dirname = dirname
        self.get_columns()

    def get_columns(self):
        px, py, n_particles = [], [], []
        for dirpath, dirnames, filenames in os.walk(self.dirname):           
            for names in filenames:
                full_path = os.path.join(dirpath, names)
                if '.dat' in names:
                    continue
                with open(full_path, 'r') as file_unit:
                    content = file_unit.read()
                if not content:
                    print('the file is empty.')
                else:
                    lines = content.split('\n')
                    try:
                        header = int(lines[0].strip())
                    except ValueError:
                        print(f'error reading the header: {full_path}')

                lines_str = '\n'.join(lines)
                lines_float = [[float(num) for num in string.split()] for string in lines] 
                lines_str = '\n'.join([' '.join(map(str, line)) for line in lines_float]) 
                data = np.genfromtxt(io.StringIO(lines_str), usecols=(2, 3), skip_header=1, unpack=True)
                pxi, pyi = data[0], data[1]

                n_particles.append(header)
                px.append(pxi)
                py.append(pyi)
        return px, py, n_particles

    def pt_calc(self):
        pt_ = []
        px, py, n_particles = self.get_columns()
        for j in range(0, 2000): 
            for i in range(len(px[j])):
                calc = np.sqrt((px[j][i])**2 + (py[j][i])**2)
                pt_ += [calc] 
        return pt_
    
    def phi_calc(self):
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

#eosl_data = manipulation(dir1)
#pxl, pyl, particles_l = eosl_data.get_columns()
#ptl = eosl_data.pt_calc()
#phifl = eosl_data.phi_calc()

eosq_data = manipulation(dir2)
pxq, pyq, particles_q = eosq_data.get_columns()
ptq = eosq_data.pt_calc()
phifq = eosq_data.phi_calc()

#data_per_event = []
data_per_event2 = []

class events_class:

    def __init__(self, data_per_event_, num_lines_per_event, pt, phif):
        self.data_per_event_ = data_per_event_
        self.num_lines_per_event = num_lines_per_event
        self.pt = pt
        self.phif = phif
        self.event_append()

    def event_append(self):
        start = 0
        for num_lines in self.num_lines_per_event:
            Pt_event = self.pt[start:start + num_lines] 
            Phi_event = self.phif[start:start + num_lines]
            self.data_per_event_.append((Pt_event, Phi_event))
            start += num_lines
        return self.data_per_event_

    def all_events(self): 
        data_per_event_ = self.event_append()
        event_n, pt_event_n, phi_event_n = [], [], []
        for i in range(0, 2000): 
            event_i = data_per_event_[i]
            pt_event, phi_event = event_i[0], event_i[1]
            event_n += [event_i] # n --> i
            pt_event_n += [pt_event]
            phi_event_n += [phi_event]
        return event_n, pt_event_n, phi_event_n

#eosl_event_data = events_class(data_per_event, particles_l, ptl, phifl)
#images, pts, phis = eosl_event_data.all_events()

eosq_event_data = events_class(data_per_event2, particles_q, ptq, phifq)
images2, pts2, phis2 = eosq_event_data.all_events()
print('-=-=-=-=-=-=-=-')

def create_data(pts_, phis_, nOevents=2000, bin_=(50, 50), save_path=None):
    final_data = []
    for i in range(0, nOevents):

        pt = pts_[i]
        phi = phis_[i]

        range_phi = (-np.pi/12, (2 * np.pi) + (np.pi/12))
        range_pt = (min(pt)-0.1, max(pt)+1)

        hist, xedges, yedges = np.histogram2d(phi, pt, bins=bin_, range=[range_phi, range_pt]) 
        plt.imshow(hist.T, extent=[range_phi[0], range_phi[1], range_pt[0], range_pt[1]], cmap='gray', aspect='auto', origin='lower')
        plt.xlim(*range_phi)
        plt.ylim(*range_pt)
        #x_labels = [range_phi[0] + (range_phi[1] - range_phi[0]) / 3, range_phi[0] + 2 * (range_phi[1] - range_phi[0]) / 3]
        #y_labels = [range_pt[0] + (range_pt[1] - range_pt[0]) / 3, range_pt[0] + 2 * (range_pt[1] - range_pt[0]) / 3]

        #plt.xticks(x_labels)
        #plt.yticks(y_labels)
        #plt.title(f'Nº of particles in an event ({i+1})')
        #plt.xlabel('Azimuthal Angle')
        #plt.ylabel('Traversial Momentum')

        if save_path:
            filename = f'{save_path}/figure__{i+1}.png'
            plt.gcf().set_size_inches(1, 1)
            plt.savefig(filename, format='png', dpi=60) # tirei o tight e inches=0 ### DPI = BIN + 10
            #plt.savefig(filename, format='png')           
            final_data.append(filename)
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            final_data.append(image_base64)
        plt.close()

    return final_data

#data_eosl = create_data(pts, phis, save_path="D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_50")
data_eosq = create_data(pts2, phis2, save_path="D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_low_50")
