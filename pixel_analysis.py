import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

print('-=-'*10)
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

    def pt_calc(self, nOevents=2000):
        pt_ = []
        px, py, n_particles = self.get_columns()
        for j in range(0, nOevents): 
            for i in range(len(px[j])):
                calc = np.sqrt((px[j][i])**2 + (py[j][i])**2)
                pt_ += [calc] 
        return pt_
    
    def phi_calc(self, nOevents=2000):
        phif_ = []
        pi = np.pi
        px, py, n_particles = self.get_columns()
        for j in range(0, nOevents):
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
eosl_data = manipulation(dir1)
pxl, pyl, particles_l = eosl_data.get_columns()
ptl = eosl_data.pt_calc(nOevents=3)
phifl = eosl_data.phi_calc(nOevents=3)
data_per_event = []

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

    def all_events(self, nOevents=2000): 
        data_per_event_ = self.event_append()
        event_n, pt_event_n, phi_event_n = [], [], []
        for i in range(0, nOevents): 
            event_i = data_per_event_[i]
            pt_event, phi_event = event_i[0], event_i[1]
            event_n += [event_i] # n --> i
            pt_event_n += [pt_event]
            phi_event_n += [phi_event]
        return event_n, pt_event_n, phi_event_n

eosl_event_data = events_class(data_per_event, particles_l, ptl, phifl)
images, pts, phis = eosl_event_data.all_events(nOevents=3)

'''def data_plot(pts_, phis_, bin_x_m=1, bin_y_m=1, bin_=(40, 40), nOevents=1, save_path=None):
    final_data = []
    cValues = []
    for i in range(0, nOevents):

        pt = pts_[i]
        phi = phis_[i]

        range_phi = (-np.pi/12, (2 * np.pi) + (np.pi/12))
        range_pt = (min(pt)-0.1, max(pt)+1)

        hist, xedges, yedges = np.histogram2d(phi, pt, bins=bin_, range=[range_phi, range_pt]) 
        plt.imshow(hist.T, extent=[range_phi[0], range_phi[1], range_pt[0], range_pt[1]], cmap='gray', aspect='auto', origin='lower')
        plt.xlim(*range_phi)
        plt.ylim(*range_pt)
        x_labels = [range_phi[0] + (range_phi[1] - range_phi[0]) / 3, range_phi[0] + 2 * (range_phi[1] - range_phi[0]) / 3]
        y_labels = [range_pt[0] + (range_pt[1] - range_pt[0]) / 3, range_pt[0] + 2 * (range_pt[1] - range_pt[0]) / 3]

        plt.xticks(x_labels)
        plt.yticks(y_labels)
        plt.title(f'Nº of particles in an event ({i+1})')
        plt.xlabel('Azimuthal Angle')
        plt.ylabel('Traversial Momentum')

        if save_path:
            filename = f'{save_path}/figure_data_{i+1}.png'
            plt.savefig(filename, format='png')           
            final_data.append(filename)
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            final_data.append(image_base64)

        #bin_cord_x = ((bin_x_m-1)*((range_phi[1] - range_phi[0])/bin_[0]) + (bin_x_m)*((range_phi[1] - range_phi[0])/bin_[1]))/2
        #bin_cord_y = ((bin_y_m-1)*((range_pt[1] - range_pt[0])/bin_[0]) + (bin_y_m)*((range_pt[1] - range_pt[0])/bin_[1]))/2
        
        #x_index = int(round((bin_cord_x - range_phi[0]) / (range_phi[1] - range_phi[0]) * (bin_[0] - 1)))
        #y_index = int(round((bin_cord_y - range_pt[0]) / (range_pt[1] - range_pt[0]) * (bin_[1] - 1)))

        #print(f"bin cord x: {bin_cord_x}; x index: {x_index}")
        #print(f"bin cord y: {bin_cord_y}; y index: {y_index}")

        #cValue = hist.T[y_index, x_index]
        #cValue = (hist.T[y_index, x_index] / np.max(hist.T)) * 255

        cValue = round((hist.T[bin_y_m - 1, bin_x_m - 1] / np.max(hist)) * 255.0)
        #cValue = hist.T[bin_y_m-1, bin_x_m-1]
        

        cValues.append(cValue)

        plt.close()

    return final_data, cValues

def particles_plot(pts_, phis_, bin_x_m=1, bin_y_m=1, bin_=(40, 40), nOevents=1, save_path=None):
    final_data = []
    nOpoints = []
    for i in range(0, nOevents):

        pt = pts_[i]
        phi = phis_[i]

        range_phi = (-np.pi/12, (2 * np.pi) + (np.pi/12))
        range_pt = (min(pt)-0.1, max(pt)+1)

        plt.scatter(phi, pt)

        plt.xlim(*range_phi)
        plt.ylim(*range_pt)
        x_labels = [range_phi[0] + (range_phi[1] - range_phi[0]) / 3, range_phi[0] + 2 * (range_phi[1] - range_phi[0]) / 3]
        y_labels = [range_pt[0] + (range_pt[1] - range_pt[0]) / 3, range_pt[0] + 2 * (range_pt[1] - range_pt[0]) / 3]

        plt.xticks(x_labels)
        plt.yticks(y_labels)
        plt.title(f'Nº of particles in an event ({i+1})')
        plt.xlabel('Azimuthal Angle')
        plt.ylabel('Traversial Momentum')

        if save_path:
            filename = f'{save_path}/figure_particle_{i+1}.png'
            plt.savefig(filename, format='png') 
            final_data.append(filename)
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            final_data.append(image_base64)

        bin_x = ((bin_x_m-1) * (range_phi[1] - range_phi[0]) / bin_[0], bin_x_m * (range_phi[1] - range_phi[0]) / bin_[1])
        print(f"bin x: {bin_x}")
        bin_y = ((bin_y_m-1) * (range_pt[1] - range_pt[0]) / bin_[0], bin_y_m * (range_pt[1] - range_pt[0]) / bin_[1])
        print(f"bin y: {bin_y}")

        plt.close()

        #points_in_bin = np.sum((pt >= bin_x[0]) & (pt <= bin_x[1]) & (phi >= bin_y[0]) & (phi <= bin_y[1]))
        #points_in_bin = sum(1 for p, ph in zip(pt, phi) if (bin_x[0] < ph < bin_x[1]) and (bin_y[0] < p < bin_y[1]))
        
        #points_in_bin = sum(1 for (ph, p) in zip(phi, pt) if ((bin_x[0] < ph < bin_x[1]) and (bin_y[0] < p < bin_y[1])))

        points_in_bin = 0
        for p in pt:
            if (bin_y[0] < p < bin_y[1]):
                for ph in phi:        
                    if (bin_x[0] < ph < bin_x[1]):
                        #print(f'pt, phi in the bin: ({p}, {ph})')
                        points_in_bin += 1
                        print(points_in_bin)
                    else:
                        pass
            else:
                pass
        
        points_in_bin = 0
        for i in range(len(phi)):
            if (bin_x[0] < phi[i] < bin_x[1]) and (bin_y[0] < pt[i] < bin_y[1]):
                points_in_bin += 1


        nOpoints.append(points_in_bin)

    return final_data, nOpoints

f_da, cVa = data_plot(pts_=pts, phis_=phis, bin_x_m=6, bin_y_m=4, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles") # 255
f_da2, nOpa = particles_plot(pts_=pts, phis_=phis, bin_x_m=6, bin_y_m=4, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles")

f_db, cVb = data_plot(pts_=pts, phis_=phis, bin_x_m=7, bin_y_m=7, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles") # 42
f_db2, nOpb = particles_plot(pts_=pts, phis_=phis, bin_x_m=7, bin_y_m=7, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles")

f_dc, cVc = data_plot(pts_=pts, phis_=phis, bin_x_m=11, bin_y_m=2, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles") # 92
f_dc3, nOpc = particles_plot(pts_=pts, phis_=phis, bin_x_m=11, bin_y_m=2, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles")

f_dd, cVd = data_plot(pts_=pts, phis_=phis, bin_x_m=1, bin_y_m=1, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles") # 0
f_dc4, nOpd = particles_plot(pts_=pts, phis_=phis, bin_x_m=1, bin_y_m=1, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles")'''

def data_plot(pts_, phis_, bin_x_m=1, bin_y_m=1, bin_=(40, 40), nOevents=1, save_path=None):
    final_data = []
    cValues = []
    particle_counts = []

    for i in range(nOevents):
        pt = pts_[i]
        phi = phis_[i]

        range_phi = (-np.pi/12, (2 * np.pi) + (np.pi/12))
        range_pt = (min(pt)-0.1, max(pt)+1)

        hist, xedges, yedges = np.histogram2d(phi, pt, bins=bin_, range=[range_phi, range_pt]) 
        plt.imshow(hist.T, extent=[range_phi[0], range_phi[1], range_pt[0], range_pt[1]], cmap='gray', aspect='auto', origin='lower')
        plt.xlim(*range_phi)
        plt.ylim(*range_pt)
        x_labels = [range_phi[0] + (range_phi[1] - range_phi[0]) / 3, range_phi[0] + 2 * (range_phi[1] - range_phi[0]) / 3]
        y_labels = [range_pt[0] + (range_pt[1] - range_pt[0]) / 3, range_pt[0] + 2 * (range_pt[1] - range_pt[0]) / 3]

        plt.xticks(x_labels)
        plt.yticks(y_labels)
        plt.title(f'Nº of particles in an event ({i+1})')
        plt.xlabel('Azimuthal Angle')
        plt.ylabel('Traversial Momentum')

        if save_path:
            filename = f'{save_path}/figure_data_{i+1}_concfunc.png'
            plt.savefig(filename, format='png')           
            final_data.append(filename)
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            final_data.append(image_base64)

        cValue = round((hist.T[bin_y_m - 1, bin_x_m - 1] / np.max(hist)) * 255.0)
        cValues.append(cValue)

        particle_count = int(hist.T[bin_y_m - 1, bin_x_m - 1])
        particle_counts.append(particle_count)

        plt.close()

    return final_data, cValues, particle_counts

f_da, cVa, nOpa = data_plot(pts_=pts, phis_=phis, bin_x_m=6, bin_y_m=4, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles") # 255

f_db, cVb, nOpb = data_plot(pts_=pts, phis_=phis, bin_x_m=7, bin_y_m=7, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles") # 42

f_dc, cVc, nOpc = data_plot(pts_=pts, phis_=phis, bin_x_m=11, bin_y_m=2, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles") # 92

f_dd, cVd, nOpd = data_plot(pts_=pts, phis_=phis, bin_x_m=1, bin_y_m=1, bin_=(40, 40), nOevents=3,
                              save_path="D:\\Users\\mathe\\ML\\EoS\\pixel-X-num-particles") # 0

print(f"first bin color value: {cVa} | num of particles: {nOpa}")
print(f"second bin color value: {cVb} | num of particles: {nOpb}")
print(f"third bin color value: {cVc} | num of particles: {nOpc}")
print(f"0 bin color value: {cVd} | num of particles: {nOpd}")
