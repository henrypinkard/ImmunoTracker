import h5py
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# settings for exporting plots to illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# make text on figures look good
SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 28
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def read_time_calibration(time_cal_filepath):
    # read time calibration
    time_cals = []
    time_file = h5py.File(time_cal_filepath, mode='r')
    for key in time_file['DataSetInfo']['TimeInfo'].attrs.keys():
        if 'TimePoint' in key and 'Dataset' not in key and 'File' not in key:
            string = ''.join([letter.decode("utf-8") for letter in time_file['DataSetInfo']['TimeInfo'].attrs[key]])
            split1 = string.split(' ')[1].split(':')
            h = int(split1[0][-1])
            m = int(split1[1]) % 60
            s = float(split1[2]) % 60
            total_s = h * 60 * 60 + m * 60 + s
            time_cals.append(total_s)
    time_cals.sort()
    elapsed_time_s = np.array(time_cals)
    return elapsed_time_s


def get_spots(file, name):
    for folder_name in file['Scene']['Content']:
        object = file['Scene']['Content'][folder_name]
        object_name = object.attrs['Name'][0].decode('utf-8')
        if object_name == name:
            coords = object['CoordsXYZR'][:, :3]
            time_indices = np.ravel(object['Time'])
            if 'Edges' in object:
                edges = object['Edges']
            else:
                edges = None
            return coords, time_indices, edges_to_tracks(edges) if edges is not None else None


def edges_to_tracks(edges):
    # sort edges into list of list of list of all indices in a given track
    tracks = []
    for edge in edges:
        new_track = True
        for track in tracks:
            if edge[0] in track or edge[1] in track:
                track.add(edge[0])
                track.add(edge[1])
                new_track = False
        if new_track:
            tracks.append(set())
            tracks[-1].add(edge[0])
            tracks[-1].add(edge[1])
    # convert to arrays
    tracks_arrays = []
    for track in tracks:
        tracks_arrays.append(np.array(list(track)))
    return tracks_arrays

def compute_frac_within_distance(dc_coords, t_cell_coords, max_dist=500):
    #compute fraction of DCs within different distances
    distances = np.linalg.norm(t_cell_coords[:, None, :] - dc_coords, 2, axis=2)
    distance_thresholds = np.arange(0, max_dist, 10)
    frac_within_distance = np.sum(distances[:, :, None] < distance_thresholds[None, None, :], axis=1) / dc_coords.shape[0]
    mean_frac_within_distance = np.mean(frac_within_distance, axis=0) #average over T cells
    se_frac_within_distance = np.std(frac_within_distance, axis=0) / np.sqrt(frac_within_distance.shape[0])

    return distance_thresholds, mean_frac_within_distance, se_frac_within_distance



filepath25 = '/Users/henrypinkard/Desktop/imaris_analysis/25.ims'

file25 = h5py.File(filepath25, mode='r')
coordsOT1, time_indicesOT1, tracksVPD = get_spots(file25, 'VPD')
coordsXCR1, time_indicesXCR1, tracksXCR1 = get_spots(file25, 'XCR1')
#use only tp0
dc_coords_control = coordsXCR1[time_indicesXCR1 == 0]
ot1_coords_control = coordsOT1[time_indicesOT1 == 0]


filepath48 = '/Users/henrypinkard/Desktop/imaris_analysis/48.ims'
file48 = h5py.File(filepath48, mode='r')
ot1_coords_24hrs = get_spots(file48, 'RFP')[0]
ot2_coords_24hrs = get_spots(file48, 'VPD')[0]
polyclonal_coords_24hrs = get_spots(file48, 'VPD')[0]
xcr1_coords_24hrs = get_spots(file48, 'XCR1')[0]

distance_thresholds, mean_frac_within_distance_control, se_control = compute_frac_within_distance(dc_coords_control, ot1_coords_control)
distance_thresholds, mean_frac_within_distance_24hrs, se_24hrs = compute_frac_within_distance(xcr1_coords_24hrs, ot1_coords_24hrs)
distance_thresholds, mean_frac_within_distance_24hrs_polyclonal, se_24hrs_controls = compute_frac_within_distance(xcr1_coords_24hrs, polyclonal_coords_24hrs)


plt.plot(distance_thresholds, mean_frac_within_distance_control)
plt.fill_between(distance_thresholds, mean_frac_within_distance_control - se_control,
                 mean_frac_within_distance_control + se_control, alpha=0.4)
plt.plot(distance_thresholds, mean_frac_within_distance_24hrs)
plt.fill_between(distance_thresholds, mean_frac_within_distance_24hrs - se_24hrs,
                 mean_frac_within_distance_24hrs + se_24hrs, alpha=0.4)
plt.plot(distance_thresholds, mean_frac_within_distance_24hrs_polyclonal)
plt.fill_between(distance_thresholds, mean_frac_within_distance_24hrs_polyclonal - se_24hrs_controls,
                 mean_frac_within_distance_24hrs_polyclonal + se_24hrs_controls, alpha=0.4)
plt.ylabel('Fraction of detected XCR1+ DCs within distance')
plt.xlabel('Distance ($\mu$m)')
plt.legend(['OT1 (no immune response)', 'OT1 (24 post-immunization)', 'Polyconal (24 hour post-immunization)'])
plt.savefig('figures/OT1-XCR1-Localization.pdf')
plt.show()
pass