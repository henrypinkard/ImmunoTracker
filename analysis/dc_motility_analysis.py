import h5py
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import scipy.ndimage as ndi
import napari


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

def filter_xcr1_tracks(dataset_id, xcr1_xyz, xcr1_time_indices, xcr1_tracks, elapsed_time_s, min_track_duration,
                       split_by_subregion=True):
    """
    Filter tracks by miniumum duration and LN subregion
    :return:
    """
    if split_by_subregion:
        is_subcapsular, is_deep_t, is_ifr, is_b_cell_zone = generate_ln_subregion_masks(xcr1_xyz, dataset_id)

    #split tracks based on starting locations
    b_cell_zone_tracks = []; deep_t_zone_tracks = []; ifr_tracks = []; subcapsular_tracks = []
    all_filtered = []
    for track in xcr1_tracks:
        # filter tracks that arent long enough
        track_duration = elapsed_time_s[xcr1_time_indices[np.max(track)]] - elapsed_time_s[
            xcr1_time_indices[np.min(track)]]
        if track_duration < min_track_duration:
            continue
        if not split_by_subregion:
            all_filtered.append(track)
        else:
            if is_b_cell_zone[track[0]]:
                b_cell_zone_tracks.append(track)
            elif is_deep_t[track[0]]:
                deep_t_zone_tracks.append(track)
            elif is_ifr[track[0]]:
                ifr_tracks.append(track)
            elif is_subcapsular[track[0]]:
                subcapsular_tracks.append(track)

    if split_by_subregion:
        return subcapsular_tracks, b_cell_zone_tracks, deep_t_zone_tracks, ifr_tracks
    else:
        return all_filtered

def dc_motility_ln_subzones(imaris_file, dataset_id, num_bootstraps, time_cal_path, min_track_duration):
    """
    control dataset (25) XCR1 motility in different lymph node regions (B cell zone, Deep t Cell zone, IFR)
    :return:
    """

    # 	Deep T, IFR - Define by proximity to both T cells and cortex, B cell
    xcr1_xyz, xcr1_time_indices, xcr1_tracks = get_spots(imaris_file, 'XCR1')
    elapsed_time_s = read_time_calibration(time_cal_path)
    # if dataset_id == 48:
    #     elapsed_time_s = np.concatenate([np.array([0]), elapsed_time_s])

    subcapsular_tracks, b_cell_zone_tracks, deep_t_zone_tracks, ifr_tracks = filter_xcr1_tracks(dataset_id,
        xcr1_xyz, xcr1_time_indices, xcr1_tracks, elapsed_time_s, min_track_duration)

    legend = []
    for i, track in enumerate([deep_t_zone_tracks, ifr_tracks, subcapsular_tracks, b_cell_zone_tracks]):
        if len(track) != 0:
            motility_analysis(xcr1_xyz, track, xcr1_time_indices, elapsed_time_s, num_bootstraps=num_bootstraps)
            legend.append(['deep T', 'IFR', 'Subcapsular', 'B cell zone'][i])

    plt.legend(legend)
    plt.xlabel('Square root time ($\sqrt{s}$)')
    plt.ylabel('Displacement ($\mu$m)')
    plt.savefig('figures/DC_motility_by_subregion{}.pdf'.format(file_id))
    plt.show()



def dc_motility_vs_hev_cortex_distance(imaris_file, file_id, time_cal_path, num_bootstraps, min_track_duration, do_hev=True):
    hev_cortex_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/with_surface_masks/distance_transforms.h5',
                                mode='r')
    cell_measurements_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/density_maps.hdf', 'r')
    elapsed_time_s = read_time_calibration(time_cal_path)
    # if file_id == 48:
    #     elapsed_time_s = np.concatenate([np.array([0]), elapsed_time_s])

    # 	Deep T, IFR - Define by proximity to both T cells and cortex, B cell
    xcr1_xyz, xcr1_time_indices, xcr1_tracks = get_spots(imaris_file, 'XCR1')

    target_shape = np.array([900, 900, 900])
    cortex_dist = hev_cortex_file['25_cortex']
    cortex_dist_resampled = ndi.zoom(cortex_dist, np.array(target_shape) / np.array(cortex_dist.shape))
    if do_hev:
        hev_dist = hev_cortex_file['25_hevs']
        hev_dist_resampled = ndi.zoom(hev_dist, np.array(target_shape) / np.array(hev_dist.shape))

    # convert into mask coordinates
    xrc1_coords_resampled = xcr1_xyz / np.array([0.555, 0.555, 4]) / np.flip(np.array(cortex_dist.shape)) * target_shape
    zyx_coords = np.flip(xrc1_coords_resampled.astype(np.int), axis=1)
    cell_cortex_dists = cortex_dist_resampled[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]
    if do_hev:
        cell_hev_dists = hev_dist_resampled[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]

    def compute_and_plot(filtered_tracks, distance):
        _, motility = compute_displacements(xcr1_xyz, filtered_tracks, xcr1_time_indices, elapsed_time_s)

        # plt.plot(density, motility, '.') #plot raw data

        # Bootstrap + Loess
        bootstrap_predctions = []
        query_points = np.linspace(0, np.max(distance), 10).astype(np.float)
        for bootstrap_iter in range(num_bootstraps):
            print('Bootstrap iteration {}'.format(bootstrap_iter))
            indices = np.random.randint(0, distance.shape[0], size=distance.shape[0])
            distance_to_structure_resampled = distance[indices]
            motility_resampled = motility[indices]
            query_pred = lowess(distance_to_structure_resampled, motility_resampled, query_points, alpha=0.3, zero_bias=True)
            bootstrap_predctions.append(query_pred)
        predictions = np.stack(bootstrap_predctions, axis=0)

        plot_bootstrapped(query_points, predictions)

    # #look at location of most motile ones
    # _, motility = compute_displacements(xcr1_xyz, xcr1_tracks, xcr1_time_indices, elapsed_time_s_25)
    # motile_coords = np.array([zyx_coords[track[0]] for index, track in enumerate(tracks_ifr) if motility[index] > 0.05])
    # non_motile_coords = np.array([zyx_coords[track[0]] for index, track in enumerate(tracks_ifr) if motility[index] <= 0.05])
    # #visualize distance map + coords
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(np.array(dc_density))
    #     viewer.add_points(motile_coords)
    #     viewer.add_points(non_motile_coords)

    filtered_tracks = []
    for track in xcr1_tracks:
        track_duration = elapsed_time_s[xcr1_time_indices[np.max(track)]] - elapsed_time_s[
            xcr1_time_indices[np.min(track)]]
        if track_duration < min_track_duration:
            continue
        filtered_tracks.append(track)

    compute_and_plot(filtered_tracks, cell_cortex_dists[[track[0] for track in filtered_tracks]])
    if do_hev:
        compute_and_plot(filtered_tracks, cell_hev_dists[[track[0] for track in filtered_tracks]])

    plt.xlabel('Distance to structure')
    plt.ylabel('Motility coefficient')
    if do_hev:
        plt.legend(['Cortex', 'HEVs'])
        plt.savefig('figures/dc_cortex_hev_dist_vs_motility_{}.pdf'.format(file_id))

def dc_motility_vs_static_t_dist(imaris_file, dataset_id, density_id, num_bootstraps, time_cal_path, min_track_duration):
    cell_measurements_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/density_maps.hdf', 'r')
    elapsed_time_s = read_time_calibration(time_cal_path)

    xcr1_xyz, xcr1_time_indices, xcr1_tracks = get_spots(imaris_file, 'XCR1')

    ot1_distance = cell_measurements_file['{}_static_ot1_distance'.format(density_id)]
    ot2_distance = cell_measurements_file['{}_static_ot2_distance'.format(density_id)]


    ### Assign cells to different regions
    # convert into mask coordinates
    image_size_um = get_image_shape(imaris_file) * np.array([4, 0.555, 0.555])
    xrc1_coords_resampled = xcr1_xyz / np.flip(image_size_um) * np.array(ot1_distance.shape)
    zyx_coords = np.flip(xrc1_coords_resampled.astype(np.int), axis=1)

    xcr1_static_ot1_distances = np.array(ot1_distance)[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]
    xcr1_static_ot2_distances = np.array(ot2_distance)[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]

    #visualize distance map
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(np.array(dc_density))

    filtered_tracks = filter_xcr1_tracks(
        dataset_id, xcr1_xyz, xcr1_time_indices, xcr1_tracks, elapsed_time_s, min_track_duration, split_by_subregion=False)

    def compute_and_plot(distances, filtered_tracks, alpha=0.3, num_bootstraps=100):
        _, motility = compute_displacements(xcr1_xyz, filtered_tracks, xcr1_time_indices, elapsed_time_s)
        density = distances[[track[0] for track in filtered_tracks]]

        # plt.plot(density, motility, '.') #plot raw data

        # Bootstrap + Loess
        bootstrap_predctions = []
        query_points = np.linspace(0, np.max(density), 10).astype(np.float)
        for bootstrap_iter in range(num_bootstraps):
            print('Bootstrap iteration {}'.format(bootstrap_iter))
            indices = np.random.randint(0, density.shape[0], size=density.shape[0])
            denisty_resampled = density[indices]
            motility_resampled = motility[indices]
            query_pred = lowess(denisty_resampled, motility_resampled, query_points, alpha=alpha, zero_bias=True)
            bootstrap_predctions.append(query_pred)
        predictions = np.stack(bootstrap_predctions, axis=0)

        plot_bootstrapped(query_points, predictions)

    # #look at location of most motile ones
    # _, motility = compute_displacements(xcr1_xyz, xcr1_tracks, xcr1_time_indices, elapsed_time_s_25)
    # motile_coords = np.array([zyx_coords[track[0]] for index, track in enumerate(tracks_ifr) if motility[index] > 0.05])
    # non_motile_coords = np.array([zyx_coords[track[0]] for index, track in enumerate(tracks_ifr) if motility[index] <= 0.05])
    # #visualize distance map + coords
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(np.array(dc_density))
    #     viewer.add_points(motile_coords)
    #     viewer.add_points(non_motile_coords)

    plt.figure()

    compute_and_plot(xcr1_static_ot1_distances, filtered_tracks, alpha=0.5, num_bootstraps=num_bootstraps)
    compute_and_plot(xcr1_static_ot2_distances, filtered_tracks, alpha=0.5, num_bootstraps=num_bootstraps)
    plt.xlabel('Distance to arrested cell')
    plt.ylabel('Motility coefficient')
    plt.legend(['OT1', 'OT2'])
    plt.savefig('figures/DC_motility_by_distance_to_ot1_ot2_{}.pdf'.format(file_id))
    plt.show()

#####  Control lymph node dynamic measurements
file_id = 25
imaris_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/25.ims', 'r')
time_cal_path = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/25_uncorrected.ims'
min_track_duration = 30 * 60
dc_motility_ln_subzones(imaris_file, file_id, num_bootstraps=100, time_cal_path=time_cal_path,
                        min_track_duration=min_track_duration)
# dc_motility_vs_dc_density(imaris_file, dataset_id='25', density_id='25', num_bootstraps=50,
#                      time_cal_path=time_cal_path, min_track_duration=min_track_duration, split_by_subregion=False)
dc_motility_vs_hev_cortex_distance(imaris_file, time_cal_path, num_bootstraps=100)


#####24 hour lymph node dynamic measurements
file_id = 48
imaris_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/48-49.ims', 'r')
time_cal_path = '/Users/henrypinkard/Desktop/imaris_analysis/48-49.ims'
min_track_duration = 30 * 60
dc_motility_ln_subzones(imaris_file, file_id, num_bootstraps=100, time_cal_path=time_cal_path,
                        min_track_duration=min_track_duration)
# dc_motility_vs_dc_density(imaris_file, file_id, density_id='48-49_cropped_to_48', num_bootstraps=1000,
#                                      time_cal_path=time_cal_path, min_track_duration=min_track_duration)
dc_motility_vs_hev_cortex_distance(imaris_file, file_id, time_cal_path, num_bootstraps=100, min_track_duration=min_track_duration)
dc_motility_vs_static_t_dist(imaris_file, file_id, file_id, num_bootstraps=100,
                             time_cal_path=time_cal_path, min_track_duration=min_track_duration)




# plt.figure()
# file_id = 25
# imaris_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/25.ims', 'r')
# time_cal_path = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/25_uncorrected.ims'
# min_track_duration = 30 * 60
# dc_motility_vs_dc_density(imaris_file, dataset_id='25', density_id='25', num_bootstraps=50,
#                      time_cal_path=time_cal_path, min_track_duration=min_track_duration, split_by_subregion=False)
# file_id = 21
# imaris_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/21.ims', 'r')
# time_cal_path = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/21_uncorrected.ims'
# min_track_duration = 30 * 60
# dc_motility_vs_dc_density(imaris_file, file_id, density_id='21', num_bootstraps=50, time_cal_path=time_cal_path,
#                           min_track_duration=min_track_duration, split_by_subregion=False)
# plt.legend(['Control', '5 hours post immunization'])
# plt.savefig('figures/control_vs_5_hour_dc_motility_dc_density.pdf')




# file_id = 25
# imaris_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/25.ims', 'r')
# time_cal_path = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/25_uncorrected.ims'
# min_track_duration = 30 * 60
# dc_motility_vs_hev_cortex_distance(imaris_file, file_id, time_cal_path, num_bootstraps=100,
#                                    min_track_duration=min_track_duration, do_hev=False)
#
#
# file_id = 21
# imaris_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/21.ims', 'r')
# time_cal_path = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/21_uncorrected.ims'
# dc_motility_vs_hev_cortex_distance(imaris_file, file_id, time_cal_path, num_bootstraps=100,
#                                    min_track_duration=min_track_duration, do_hev=False)
#
# plt.legend(['Control', '5 hours post immunization'])
# plt.savefig('figures/dc_cortex_dist_vs_motility_5hour_or_control.pdf')


##T cell denisty
file_id = 21
min_track_duration = 0
imaris_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/21.ims', 'r')
time_cal_path = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/21_uncorrected.ims'
dc_motility_vs_cell_density(imaris_file, file_id, density_id='21', num_bootstraps=100, split_by_subregion=False,
                            time_cal_path=time_cal_path,min_track_duration=min_track_duration, ref_cell_name='ot1',
                            distance_threshold=60)
plt.savefig('figures/5_hour_motility_by_ot1_density_60_um.pdf')

# dc_motility_vs_cell_density(imaris_file, file_id, density_id='21', num_bootstraps=100, split_by_subregion=False,
#                     time_cal_path=time_cal_path, min_track_duration=min_track_duration, ref_cell_name='ot1static',
#                     distance_threshold=60)
# plt.savefig('figures/5_hour_motility_by_ot1static_density_60_um.pdf')