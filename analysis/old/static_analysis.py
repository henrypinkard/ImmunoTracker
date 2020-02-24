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



filepath25 = '/Users/henrypinkard/Desktop/imaris_analysis/25.ims'
file25 = h5py.File(filepath25, mode='r')
coordsOT1, time_indicesOT1, tracksVPD = get_spots(file25, 'VPD')
coordsXCR1, time_indicesXCR1, tracksXCR1 = get_spots(file25, 'XCR1')
#use only tp0
xcr1_coords_control = coordsXCR1[time_indicesXCR1 == 0]
ot1_coords_control = coordsOT1[time_indicesOT1 == 0]


filepath48 = '/Users/henrypinkard/Desktop/imaris_analysis/48.ims'
file48 = h5py.File(filepath48, mode='r')
ot1_coords_24hrs = get_spots(file48, 'RFP')[0]
ot2_coords_24hrs = get_spots(file48, 'VPD')[0]
polyclonal_coords_24hrs = get_spots(file48, 'VPD')[0]
xcr1_coords_24hrs = get_spots(file48, 'XCR1')[0]

distance_thresholds, mean_frac_within_distance_control_ot1, se_control_ot1 = compute_frac_within_distance(xcr1_coords_control, ot1_coords_control)
distance_thresholds, mean_frac_within_distance_24hrs_ot1, se_24hrs_ot1 = compute_frac_within_distance(xcr1_coords_24hrs, ot1_coords_24hrs)
distance_thresholds, mean_frac_within_distance_24hrs_polyclonal, se_24hrs_polyclonal = compute_frac_within_distance(xcr1_coords_24hrs, polyclonal_coords_24hrs)

distance_thresholds, mean_frac_within_distance_control_xcr1s, se_control_xcr1 = compute_frac_within_distance(xcr1_coords_control, xcr1_coords_control)
distance_thresholds, mean_frac_within_distance_24hrs_xcr1s, se_24hrs_xcr1 = compute_frac_within_distance(xcr1_coords_24hrs, xcr1_coords_24hrs)


plt.plot(distance_thresholds, mean_frac_within_distance_control_ot1)
plt.fill_between(distance_thresholds, mean_frac_within_distance_control_ot1 - se_control_ot1,
                 mean_frac_within_distance_control_ot1 + se_control_ot1, alpha=0.4)
plt.plot(distance_thresholds, mean_frac_within_distance_24hrs_ot1)
plt.fill_between(distance_thresholds, mean_frac_within_distance_24hrs_ot1 - se_24hrs_ot1,
                 mean_frac_within_distance_24hrs_ot1 + se_24hrs_ot1, alpha=0.4)
plt.plot(distance_thresholds, mean_frac_within_distance_24hrs_polyclonal)
plt.fill_between(distance_thresholds, mean_frac_within_distance_24hrs_polyclonal - se_24hrs_polyclonal,
                 mean_frac_within_distance_24hrs_polyclonal + se_24hrs_polyclonal, alpha=0.4)
plt.ylabel('Fraction of detected XCR1+ DCs within distance')
plt.xlabel('Distance ($\mu$m)')
plt.legend(['OT1 (no immune response)', 'OT1 (24 post-immunization)', 'Polyconal (24 hour post-immunization)'])
plt.savefig('figures/OT1-XCR1-Localization.pdf')


plt.figure()
plt.plot(distance_thresholds, mean_frac_within_distance_control_xcr1s)
plt.fill_between(distance_thresholds, mean_frac_within_distance_control_xcr1s - se_control_xcr1,
                 mean_frac_within_distance_control_xcr1s + se_control_xcr1, alpha=0.4)
plt.plot(distance_thresholds, mean_frac_within_distance_24hrs_xcr1s)
plt.fill_between(distance_thresholds, mean_frac_within_distance_24hrs_xcr1s - se_24hrs_xcr1,
                 mean_frac_within_distance_24hrs_xcr1s + se_24hrs_xcr1, alpha=0.4)
plt.ylabel('Fraction of detected XCR1+ DCs within distance')
plt.xlabel('Distance ($\mu$m)')
plt.legend(['control', '24 hour post-immunization'])
plt.savefig('figures/DC-DC-Localization.pdf')
plt.show()



###### Bar graph of Cell locations at 0 and 24 hours

# is_subcapsular_control, is_deep_t_control, is_ifr_control, is_b_cell_zone_control = \
#     generate_ln_subregion_masks(xcr1_coords_control, '25')
# all_masks_control = np.stack([is_subcapsular_control, is_deep_t_control, is_ifr_control, is_b_cell_zone_control], axis=1)
# proportion_control = all_masks_control.astype(np.int) / is_subcapsular_control.size
# mean_proportion_control = np.sum(proportion_control, axis=0)
# stde_control = mean_proportion_control * (1 - mean_proportion_control) / np.sum(all_masks_control, axis=0)
# mean_number_control = np.sum(all_masks_control, axis=0)
#
# is_subcapsular_24, is_deep_t_24, is_ifr_24, is_b_cell_zone_24 = \
#     generate_ln_subregion_masks(xcr1_coords_24hrs, '48')
# all_masks_24 = np.stack([is_subcapsular_24, is_deep_t_24, is_ifr_24, is_b_cell_zone_24], axis=1)
# proportion_24 = all_masks_24.astype(np.int) / is_subcapsular_24.size
# mean_proportion_24 = np.sum(proportion_24, axis=0)
# stde_24 = mean_proportion_24 * (1 - mean_proportion_24) / np.sum(all_masks_24, axis=0)
# mean_number_24 = np.sum(all_masks_24, axis=0)
#
#
# width = 0.1
# labels = ('Subsapsular', 'Deep T cell zone', 'IFR', 'B cell zone')
# x = np.arange(len(labels))
# fig, ax = plt.subplots()
# for i in range(4):
#     ax.bar(np.array([0, 1]) + width*i, [mean_number_control[i], mean_number_24[i]], width=width, label=labels[i])
# plt.ylabel('# XCR1+ cells detected in region')
# plt.xticks(np.array([0, 1]) + 2*width, ('Control', '24 hrs post-immunization'))
# # plt.yticks(np.arange(0, 81, 10))
# plt.legend(labels)
# plt.savefig('figures/DCs_by_subregion_bar_graph.pdf')
# plt.show()


def measure_distances_and_densities(file_id, cell_xyz, distance):
    hev_cortex_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/with_surface_masks/distance_transforms.h5',
                                    mode='r')
    cell_measurements_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/density_maps.hdf', 'r')

    target_shape = np.array([600, 600, 600])
    cortex_dist = hev_cortex_file['25_cortex']
    hev_dist = hev_cortex_file['25_hevs']
    cortex_dist_resampled = ndi.zoom(cortex_dist, np.array(target_shape) / np.array(cortex_dist.shape))
    hev_dist_resampled = ndi.zoom(hev_dist, np.array(target_shape) / np.array(hev_dist.shape))
    dc_density = cell_measurements_file['{}_xcr1_within_'.format(file_id, distance)]

    # convert into mask coordinates
    xrc1_coords_resampled = cell_xyz / np.array([0.555, 0.555, 4]) / np.flip(np.array(cortex_dist.shape)) * target_shape
    zyx_coords = np.flip(xrc1_coords_resampled.astype(np.int), axis=1)
    cell_cortex_dists = cortex_dist_resampled[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]
    cell_hev_dists = hev_dist_resampled[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]


    xrc1_coords_resampled = cell_xyz / np.array([0.555, 0.555, 4]) / np.flip(np.array(cortex_dist.shape)) * \
                            np.array(dc_density.shape)
    zyx_coords = np.flip(xrc1_coords_resampled.astype(np.int), axis=1)
    dc_densities = np.array(dc_density)[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]
    return dc_densities, cell_cortex_dists, cell_hev_dists


def bootstrap_n_plot(x, y, n_query_points=40, num_bootstraps=100, alpha=0.2, zero_bias=True):
    # Bootstrap + Loess
    bootstrap_predctions = []
    query_points = np.linspace(0, np.max(x), n_query_points).astype(np.float)
    for bootstrap_iter in range(num_bootstraps):
        print('Bootstrap iteration {}'.format(bootstrap_iter))
        indices = np.random.randint(0, x.shape[0], size=x.shape[0])
        x_resampled = x[indices]
        y_resampled = y[indices]
        query_pred = lowess(x_resampled, y_resampled, query_points, alpha=alpha,
                            zero_bias=zero_bias, weight_fn='gaussian', window_sigma=0.15)
        bootstrap_predctions.append(query_pred)
    predictions = np.stack(bootstrap_predctions, axis=0)

    plot_bootstrapped(query_points, predictions)


density_control, cortex_dist_control, hev_dist_control = measure_distances_and_densities('25', xcr1_coords_control)
density_24, cortex_dist_24, hev_dist_24 = measure_distances_and_densities('48-49_cropped_to_48', xcr1_coords_24hrs)

alpha = 0.15

plt.figure()
bootstrap_n_plot(cortex_dist_control, density_control, num_bootstraps=50, alpha=alpha)
bootstrap_n_plot(cortex_dist_24, density_24, num_bootstraps=50, alpha=alpha)
plt.xlabel('Distance to cortex')
plt.ylabel('# XCR1+ cells within 150 um')
plt.legend(['Control', '24 hours post infection'])
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.savefig('figures/DC_density_vs_dist_to_cortex.pdf')

plt.figure()
bootstrap_n_plot(hev_dist_control, density_control, num_bootstraps=50, alpha=alpha)
bootstrap_n_plot(hev_dist_24, density_24, num_bootstraps=50, alpha=alpha)
plt.xlabel('Distance to HEVs')
plt.ylabel('# XCR1+ cells within 150 um')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.legend(['Control', '24 hours post infection'])
plt.savefig('figures/DC_density_vs_dist_to_hev.pdf')

# #raw data
# plt.figure()
# plt.plot(hev_dist_24, density_24, '.')

