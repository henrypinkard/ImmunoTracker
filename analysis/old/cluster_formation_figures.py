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

from util import *


filepath21 = '/Users/henrypinkard/Desktop/imaris_analysis/21.ims'
file21 = h5py.File(filepath21, mode='r')
time_cal_filepath21 = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/21_uncorrected.ims'
elapsed_time_s21 = read_time_calibration(time_cal_filepath21)

vpd_static, vpd_time_indices, _ = get_spots(file21, 'VPD Static')
vpd, vpd_time_indices, _ = get_spots(file21, 'VPD')
# xcr1, xcr1_time_indices, xcr1_tracks = get_spots(file21, 'XCR1')
xcr1, xcr1_time_indices, xcr1_tracks = get_spots(file21, 'XCR1 Long')


distance_within = 15
num_query = 20
sqrt_time = True
num_bootstraps = 50

#find DCs near static T cells
distances_to_static = np.linalg.norm(xcr1[:, None, :] - vpd_static, 2, axis=2)
min_distances_to_static = np.min(distances_to_static, axis=1)
static_t_cell_mask = min_distances_to_static < distance_within
static_t_cell_indices = np.where(static_t_cell_mask)[0]
#find DCs near non-static T cells
distances_to_any = np.linalg.norm(xcr1[:, None, :] - vpd, 2, axis=2)
min_distances_to_any = np.min(distances_to_any, axis=1)
other_t_cell_mask = np.logical_and(min_distances_to_any < distance_within, np.logical_not(static_t_cell_mask))
other_t_cell_indices = np.where(other_t_cell_mask)[0]

close_to_static_t_tracks = []
close_to_other_t_tracks = []
for track in xcr1_tracks:
    if np.all(np.isin(track, static_t_cell_indices)):
        close_to_static_t_tracks.append(track)
    elif np.all(np.isin(track, other_t_cell_indices)):
        close_to_other_t_tracks.append(track)

data_static = compute_displacements_over_time(xcr1, close_to_static_t_tracks, xcr1_time_indices, elapsed_time_s21)
data_other = compute_displacements_over_time(xcr1, close_to_other_t_tracks, xcr1_time_indices, elapsed_time_s21)

plt.plot((data_static[:, 0]), data_static[:, 1], '.') #plot raw data
plt.plot((data_other[:, 0]), data_other[:, 1], '.') #plot raw data

motility_coeffs_static = data_static[:, 1] / np.sqrt(data_static[:, 0])
motility_coeffs_other = data_other[:, 1] / np.sqrt(data_other[:, 0])
bins = np.linspace(0, np.max(np.concatenate([motility_coeffs_static, motility_coeffs_other])), 20)
plt.figure()
plt.hist(motility_coeffs_static, bins, density=True, histtype='step', fill=False)
plt.hist(motility_coeffs_other, bins, density=True, histtype='step', fill=False)
plt.legend(['Near static OT1s', 'Near non-static T cells'])
plt.ylabel('Probability density')
plt.xlabel('Motility coefficent ($\mu$m/$\sqrt{s}$)')
# plt.show()
plt.savefig('figures/cluster_formation_hisotgrams.pdf')

# LOWESS fit with Bootstrapped error bars
query_points_0hours = np.linspace(0, np.max(np.sqrt(data_static[:, 0]) if sqrt_time else data_static[:, 0]),
                                  num_query)
query_points_24hours = np.linspace(0, np.max(np.sqrt(data_other[:, 0]) if sqrt_time else data_other[:, 0]),
                                   num_query)
bootstrap_data_0hours = bootstrap(data_static, query_points_0hours, num_bootstraps=num_bootstraps, sqrt=sqrt_time,
                                  alpha=0.4)
bootstrap_data_24hours = bootstrap(data_other, query_points_24hours, num_bootstraps=num_bootstraps, sqrt=sqrt_time,
                                   alpha=0.4)
plt.figure()
plot_bootstrapped(query_points_0hours, bootstrap_data_0hours)
plot_bootstrapped(query_points_24hours, bootstrap_data_24hours)
plt.legend(['Near static OT1s', 'Near non-static T cells'])
plt.xlabel('Square root time ($\sqrt{s}$)')
plt.ylabel('Displacement ($\mu$m)')
plt.savefig('figures/cluster_formation_motility.pdf')
plt.show()

pass