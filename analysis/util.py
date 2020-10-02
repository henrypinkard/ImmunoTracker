import h5py
import numpy as np
import napari
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing.pool import ThreadPool
from joblib import Memory
memory = Memory('Users/henrypinkard/joblibcache', verbose=0)

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

import scipy.ndimage as ndi

def get_image_shape(imaris_file):
    x = int(imaris_file['DataSetInfo']['Image'].attrs['X'].tostring().decode('utf-8'))
    y = int(imaris_file['DataSetInfo']['Image'].attrs['Y'].tostring().decode('utf-8'))
    z = int(imaris_file['DataSetInfo']['Image'].attrs['Z'].tostring().decode('utf-8'))
    return np.array([x, y, z])

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
    for i, edge in enumerate(edges):
        print('Edge {} of {}\r'.format(i, len(edges)), end='')
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
        track_arr = np.array(list(track))
        track_arr = np.sort(track_arr)
        tracks_arrays.append(track_arr)
    return tracks_arrays

def compute_frac_within_distance(dc_coords, t_cell_coords, max_dist=500):
    #compute fraction of DCs within different distances
    distances = np.linalg.norm(t_cell_coords[:, None, :] - dc_coords, 2, axis=2)
    distance_thresholds = np.arange(0, max_dist, 10)
    frac_within_distance = np.sum(distances[:, :, None] < distance_thresholds[None, None, :], axis=1) / dc_coords.shape[0]
    mean_frac_within_distance = np.mean(frac_within_distance, axis=0) #average over T cells
    se_frac_within_distance = np.std(frac_within_distance, axis=0) / np.sqrt(frac_within_distance.shape[0])

    return distance_thresholds, mean_frac_within_distance, se_frac_within_distance

def compute_displacements(coords, tracks, time_indices, elapsed_time_s):
    displacements = []
    total_times = []
    for track in tracks:
        ts = time_indices[track]
        min_time = elapsed_time_s[min(ts)]
        max_time = elapsed_time_s[max(ts)]
        total_times.append(max_time - min_time)
        # get xyz indices corresponding to min and max time
        min_index = track[np.argmin(ts)]
        max_index = track[np.argmax(ts)]
        start_xyz = coords[min_index]
        end_xyz = coords[max_index]
        displacements.append(np.linalg.norm(end_xyz - start_xyz, 2))
    d = np.array(displacements)
    t = np.array(total_times / 60)
    m = d ** 2 / (6 * t)
    return t, m


def compute_displacements_over_time(coords, tracks, time_indices, elapsed_time_s, window_tracks=True):
    time_displacement_tuples = []
    for track in tracks:
        sorted_track = np.sort(track)
        if window_tracks:
            for start_index in range(sorted_track.size - 1):
                for end_index in range(start_index + 1, sorted_track.size):
                    start_spot_index = sorted_track[start_index]
                    end_spot_index = sorted_track[end_index]
                    # num_time_steps = time_indices[end_spot_index] - time_indices[start_spot_index]
                    track_time = elapsed_time_s[time_indices[end_spot_index]] - elapsed_time_s[
                        time_indices[start_spot_index]]
                    track_displacement = np.linalg.norm(coords[end_spot_index] - coords[start_spot_index], 2)
                    time_displacement_tuples.append((track_time, track_displacement))
        else:
            start_index = 0
            end_index = sorted_track.size - 1

            start_spot_index = sorted_track[start_index]
            end_spot_index = sorted_track[end_index]
            # num_time_steps = time_indices[end_spot_index] - time_indices[start_spot_index]
            track_time = elapsed_time_s[time_indices[end_spot_index]] - elapsed_time_s[
                time_indices[start_spot_index]]
            track_displacement = np.linalg.norm(coords[end_spot_index] - coords[start_spot_index], 2)
            time_displacement_tuples.append((track_time, track_displacement))
    return np.array(time_displacement_tuples)

def lowess(x, y, query_points, alpha, zero_bias=False, weight_fn='tricubic', window_sigma=0.5):
    predicted_vals = []
    for query_point in query_points:
        # find alpha fraction of closest points
        distances = np.abs(x - query_point)
        mask = np.argsort(distances)[:int(alpha * x.size)]
        # train regressor
        A = x[mask][:, None]
        b = y[mask][:, None]

        if np.max(mask) == int(alpha * x.size) - 1 or zero_bias:  # Don't have intercept term for first one
            A = np.concatenate([A, np.ones_like(A)], axis=1)  # add bias term

        dist = np.abs((x[mask] - query_point))
        if np.max(dist) != 0:
            dist /= np.max(dist)
        if weight_fn == 'tricubic':
            weighted = (1 - dist ** 3) ** 3
        elif weight_fn == 'gaussian':
            weighted = np.exp( -0.5*(dist / window_sigma) **2 )
        else:
            raise Exception('unknown weight fn')
        w = np.sqrt(np.diag(weighted))
        # rewight to do weighted least sqaures
        b_w = np.dot(w, b)
        A_w = np.dot(w, A)
        # solve with least squares solver
        coeffs = np.linalg.lstsq(A_w, b_w)[0].T
        if coeffs.size == 1:
            coeffs = np.concatenate([coeffs, np.array([[0]])], axis=1)
        # predict value based on this regressor
        predicted_vals.append(coeffs[:, 0] * query_point + coeffs[:, 1])
    return np.ravel(np.array(predicted_vals))

def plot_bootstrapped(query_points, bootstrap_data, normalize=False):
    error_lo = np.percentile(bootstrap_data, 2.5, axis=0)
    error_hi = np.percentile(bootstrap_data, 97.5, axis=0)
    mean = np.mean(bootstrap_data, axis=0)
    if normalize:
        error_hi /= mean[0]
        error_lo /= mean[0]
        mean /= mean[0]
    plt.plot(query_points, mean, '-')
    plt.fill_between(query_points, error_lo, error_hi, alpha=0.4)

# Locally weighted regression estiamte. Cleveland, W.S. (1979) “Robust Locally Weighted Regression and Smoothing Scatterplots”. Journal of the American Statistical Association 74 (368): 829-836.
# Bootstrap estimate: https://thestatsgeek.com/2013/07/02/the-miracle-of-the-bootstrap/

def bootstrap_and_plot(x, y, n_query_points=40, num_bootstraps=100, alpha=0.4, zero_bias=False, window_sigma=0.5,
              weight_fn='tricubic', normalize=False):
    # Bootstrap + Loess
    bootstrap_predctions = []
    query_points = np.linspace(np.min(x) if zero_bias else 0, np.max(x), n_query_points).astype(np.float)
    for bootstrap_iter in range(num_bootstraps):
        print('Bootstrap iteration {} of {}     \r'.format(bootstrap_iter+1, num_bootstraps), end='')
        indices = np.random.randint(0, x.shape[0], size=x.shape[0])
        x_resampled = x[indices]
        y_resampled = y[indices]
        query_pred = lowess(x_resampled, y_resampled, query_points, alpha=alpha,
                            zero_bias=zero_bias, weight_fn=weight_fn, window_sigma=window_sigma)
        bootstrap_predctions.append(query_pred)
    predictions = np.stack(bootstrap_predctions, axis=0)
    plot_bootstrapped(query_points, predictions, normalize=normalize)


def generate_ln_subregion_masks(imaris_file, xyz_coords, dataset_id):

    cell_measurements_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/density_maps.hdf', 'r')

    cortex_dist_map_name = '{}_cortex'.format(dataset_id)
    t_cell_zone_mask = cell_measurements_file['{}_t_cell_mask'.format(dataset_id)]

    #resample to same resolution t cell mask is at
    cortex_dist_resampled = resmaple_dist_map(cortex_dist_map_name, np.array(t_cell_zone_mask.shape))

    #Define IFR as region in T cell mask within a certain distance from cortex
    deep_t_cutoff = 190
    subcapsular_cutoff = 20
    ifr_mask = np.copy(t_cell_zone_mask)
    deep_t_mask = np.copy(t_cell_zone_mask)
    ifr_mask[cortex_dist_resampled < deep_t_cutoff] = 0
    deep_t_mask[cortex_dist_resampled >= deep_t_cutoff] = 0
    subcapsular_mask = np.zeros_like(t_cell_zone_mask)
    subcapsular_mask[cortex_dist_resampled < subcapsular_cutoff] = 1
    ifr_mask[subcapsular_mask] = 0
    deep_t_mask[subcapsular_mask] = 0

    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(ifr_mask, colormap='blue')
    #     viewer.add_image(deep_t_mask, colormap='green')
    #     viewer.add_image(subcapsular_mask, colormap='red')

    ### Assign cells to different regions
    # convert into mask coordinates
    xrc1_coords_resampled = xyz_coords / np.array([0.555, 0.555, 4]) / get_image_shape(imaris_file) * \
                            np.array(t_cell_zone_mask.shape)
    zyx_coords = np.flip(xrc1_coords_resampled.astype(np.int), axis=1)
    # create mask for cells in different regions
    is_subcapsular = subcapsular_mask[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]
    is_deep_t = np.logical_and(deep_t_mask[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]],
                               np.logical_not(is_subcapsular))
    is_ifr = np.logical_and(ifr_mask[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]],
                            np.logical_not(is_subcapsular))
    xcr1_cortex_dists = cortex_dist_resampled[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]
    is_b_cell_zone = np.logical_and(np.logical_and(np.logical_not(
        np.logical_or(is_deep_t, is_ifr)), xcr1_cortex_dists <= deep_t_cutoff), np.logical_not(is_subcapsular))

    return is_subcapsular, is_deep_t, is_ifr, is_b_cell_zone
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_points(zyx_coords[is_deep_t], size=0.6, n_dimensional=True, face_color='blue')
    #     viewer.add_points(zyx_coords[is_ifr], size=0.6, n_dimensional=True, face_color='red')
    #     viewer.add_points(zyx_coords[is_b_cell_zone], size=0.6, n_dimensional=True, face_color='green')
    #     viewer.add_points(zyx_coords[is_subcapsular], size=0.6, n_dimensional=True, face_color='yellow')
    #
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(t_cell_zone_mask)
    #     viewer.add_image(cortex_dist_resampled)
    #     viewer.add_points(zyx_coords, size=0.6, n_dimensional=True, face_color='blue')

def compute_cell_density(distance_threshold=100, file_id=25, num_samples=150, cell_name='XCR1', split_by_time=False,
                         timepoints=None):
    """
    Compute number of DCs within threshold um at each pixel in resampled volume of the image
    """

    root_path = '/Users/henrypinkard/Desktop/imaris_analysis/'
    filepath = root_path + '{}.ims'.format(file_id)
    with h5py.File(filepath, mode='r') as file:
        cell_coords, time_indices, tracks = get_spots(file, cell_name)
        image_shape = np.flip(get_image_shape(file))

    if not split_by_time:
        max_time_index = 1
    else:
        max_time_index = int(np.max(time_indices) + 1)
    with h5py.File(root_path + 'density_maps.hdf', 'a') as out_file:
        for time_index in range(max_time_index) if timepoints is None else timepoints:
            if '{}_{}_within_{}_tp_{}'.format(file_id, cell_name.lower(),
                                                                        distance_threshold, time_index) in out_file:
                print('tp {} already present, skipping to next'.format(time_index))
                continue
            coords_at_tp = cell_coords[time_indices == time_index]
            image_size_um = np.array(image_shape) * np.array([4, 0.555, 0.555])
            print('\n\n\n\n\nComputing density {} time index {}'.format(file_id, time_index))

            zz, yy, xx = np.meshgrid(np.linspace(0, image_size_um[0], num_samples),
                                     np.linspace(0, image_size_um[1], num_samples),
                                     np.linspace(0, image_size_um[2], num_samples), indexing='ij')
            all_coords = np.stack([np.ravel(xx), np.ravel(yy), np.ravel(zz)], axis=1)
            batch_size = 5000
            #iterate through batches to save memory
            def process_batch(batch_index):
                print('{} of {}'.format(batch_index, all_coords.shape[0] // batch_size))
                distances_batch = np.linalg.norm(coords_at_tp[:, None, :] - all_coords[batch_index * batch_size:
                                            (batch_index + 1) * batch_size], axis=2)
                return np.sum(distances_batch < distance_threshold, axis=0)
            pool = ThreadPool(8)
            counts = pool.map(process_batch, range(all_coords.shape[0] // batch_size + 1))
            all_counts = np.concatenate(counts, axis=0)

            count_image = np.reshape(all_counts, 3*[num_samples])
            out_file.create_dataset(name='{}_{}_within_{}_tp_{}'.format(file_id, cell_name.lower(),
                                                                        distance_threshold, time_index), data=count_image)


def filter_tracks(imaris_file, dataset_id, cell_xyz, time_indices, tracks, elapsed_time_s, min_track_duration,
                  split_by_subregion=True):
    """
    Filter tracks by miniumum duration and LN subregion
    :return:
    """
    if split_by_subregion:
        is_subcapsular, is_deep_t, is_ifr, is_b_cell_zone = generate_ln_subregion_masks(imaris_file, cell_xyz, dataset_id)

    # split tracks based on starting locations
    b_cell_zone_tracks = [];
    deep_t_zone_tracks = [];
    ifr_tracks = [];
    subcapsular_tracks = []
    all_filtered = []
    for track in tracks:
        # filter tracks that arent long enough
        track_duration = elapsed_time_s[time_indices[np.max(track)]] - elapsed_time_s[
            time_indices[np.min(track)]]
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


def dc_motility_ln_subzones(imaris_files, dataset_ids, time_cal_paths, num_bootstraps, min_track_duration, alpha=0.4):
    """
     XCR1 motility in different lymph node regions (B cell zone, Deep t Cell zone, IFR)
    :return:
    """
    imaris_file_control, imaris_file_24 = imaris_files
    dataset_id_control, dataset_id_24 = dataset_ids
    time_cal_path_control, time_cal_path_24 = time_cal_paths

    # Deep T, IFR - Define by proximity to both T cells and cortex, B cell
    xcr1_xyz_control, xcr1_time_indices_control, xcr1_tracks_control = get_spots(imaris_file_control, 'XCR1')
    elapsed_time_s_control = read_time_calibration(time_cal_path_control)
    subcapsular_tracks_control, b_cell_zone_tracks_control, deep_t_zone_tracks_control, ifr_tracks_control = filter_tracks(
        imaris_file_control, dataset_id_control, xcr1_xyz_control, xcr1_time_indices_control, xcr1_tracks_control, elapsed_time_s_control, min_track_duration)

    xcr1_xyz_24, xcr1_time_indices_24, xcr1_tracks_24 = get_spots(imaris_file_24, 'XCR1')
    elapsed_time_s_24 = read_time_calibration(time_cal_path_24)
    subcapsular_tracks_24, b_cell_zone_tracks_24, deep_t_zone_tracks_24, ifr_tracks_24 = filter_tracks(
        imaris_file_control, dataset_id_24, xcr1_xyz_24, xcr1_time_indices_24, xcr1_tracks_24, elapsed_time_s_24,
        min_track_duration)

    plt.figure()
    for i, track_set in enumerate([(deep_t_zone_tracks_control, deep_t_zone_tracks_24),
                               (ifr_tracks_control, ifr_tracks_24),
                               (subcapsular_tracks_control, subcapsular_tracks_24)]):

        displacement_time_data = compute_displacements_over_time(xcr1_xyz_control, track_set[0], xcr1_time_indices_control, elapsed_time_s_control)
        # plt.plot(np.sqrt(data_ifr_zone[:, 0]), data_ifr_zone[:, 1], '.')  # plot raw data
        bootstrap_and_plot(np.sqrt(displacement_time_data[:, 0]), displacement_time_data[:, 1], n_query_points=10,
                           num_bootstraps=num_bootstraps, alpha=alpha, zero_bias=False)
        #24 hour post
        displacement_time_data = compute_displacements_over_time(xcr1_xyz_24, track_set[1],
                                                                 xcr1_time_indices_24, elapsed_time_s_24)
        # plt.plot(np.sqrt(data_ifr_zone[:, 0]), data_ifr_zone[:, 1], '.')  # plot raw data
        bootstrap_and_plot(np.sqrt(displacement_time_data[:, 0]), displacement_time_data[:, 1], n_query_points=10,
                           num_bootstraps=num_bootstraps, alpha=alpha, zero_bias=False)

        plt.xlabel('Square root time ($\sqrt{s}$)')
        plt.ylabel('Displacement ($\mu$m)')
    plt.legend(['Deep T control', 'Deep T 24 hours post-infection', 'IFR control', 'IFR 24 hours post-infection',
                'Subcapsular control', 'Subcapsular 24 hours post-infection'])
    # plt.title(['deep_t', 'ifr', 'subcap'][i])

    plt.savefig('figures/DC_motility_by_subregion_{}.pdf'.format(['deep_t', 'ifr', 'subcap'][i]))


def motility_vs_hev_cortex_distance(file, cell_population, elapsed_time_s,
                                       num_bootstraps, min_track_duration, distance_file_index, do_hev=True,
                                    do_cortex=True, n_query_points=40, alpha=0.4, window_sigma=0.5,
                                    weight_fn='tricubic', plot_raw=False):

    spots_xyz, spots_time_indices, tracks = get_spots(file, cell_population)
    #### load distances and interpolate/resample to a better resolution
    target_shape = np.array([900, 900, 900])
    if do_cortex:
        cortex_dist_name = '{}_cortex'.format(distance_file_index)
        cortex_dist_resampled = resmaple_dist_map(cortex_dist_name, target_shape)
    if do_hev:
        hev_dist_name = '{}_hevs'.format(distance_file_index)
        hev_dist_resampled = resmaple_dist_map(hev_dist_name, target_shape)

    # convert into mask coordinates
    xrc1_coords_resampled = spots_xyz / np.array([0.555, 0.555, 4]) / get_image_shape(file) * target_shape
    zyx_coords = np.flip(xrc1_coords_resampled.astype(np.int), axis=1)
    if do_cortex:
        cell_cortex_dists = cortex_dist_resampled[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]
    if do_hev:
        cell_hev_dists = hev_dist_resampled[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]


    ##### filter out tracks below a minimum duration
    filtered_tracks = []
    for track in tracks:
        track_duration = elapsed_time_s[spots_time_indices[np.max(track)]] - elapsed_time_s[
            spots_time_indices[np.min(track)]]
        if track_duration < min_track_duration:
            continue
        filtered_tracks.append(track)

    ##Take distances for only the filtered tracks
    if do_cortex:
        cortex_dists_filtered = cell_cortex_dists[[track[0] for track in filtered_tracks]]
    else:
        cortex_dists_filtered = None
    if do_hev:
        hev_dists_filtered = cell_hev_dists[[track[0] for track in filtered_tracks]]
    _, motility = compute_displacements(spots_xyz, filtered_tracks, spots_time_indices, elapsed_time_s)

    return cortex_dists_filtered, hev_dists_filtered, motility
    # #####  make plot
    # _, motility = compute_displacements(spots_xyz, filtered_tracks, spots_time_indices, elapsed_time_s)
    # if do_cortex:
    #     x = cell_cortex_dists[[track[0] for track in filtered_tracks]]
    #     if plot_raw:
    #         plt.plot(x, motility,'.')
    #     else:
    #         bootstrap_and_plot(x, motility,
    #                    n_query_points=n_query_points, num_bootstraps=num_bootstraps, alpha=alpha, zero_bias=True,
    #                    window_sigma=window_sigma, weight_fn=weight_fn)
    # if do_hev:
    #     x = cell_hev_dists[[track[0] for track in filtered_tracks]]
    #     if plot_raw:
    #         plt.plot(x, motility,'.')
    #     else:
    #         bootstrap_and_plot(x, motility,
    #                        n_query_points=n_query_points, num_bootstraps=num_bootstraps, alpha=alpha, zero_bias=True,
    #                        window_sigma=window_sigma, weight_fn=weight_fn)
    # plt.xlabel('Distance to structure')
    # plt.ylabel('Motility coefficient')

@memory.cache
def resmaple_dist_map(name, target_shape):
    hev_cortex_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/with_surface_masks/distance_transforms.h5',
                                mode='r')
    dist = hev_cortex_file[name]
    return ndi.zoom(dist, np.array(target_shape) / np.array(dist.shape))

def measure_distances_and_densities(file_id, file, cell_xyz, distance=100):
    """
    Sample from DC density and distance maps of HEVs and cortex for a given cell pipulation
    :param file_id:
    :param cell_xyz:
    :param distance:
    :return:
    """
    cell_measurements_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/density_maps.hdf', 'r')

    target_shape = np.array([900, 900, 900])
    cortex_name = '{}_cortex'.format(file_id[:2])
    hev_name = '{}_hevs'.format(file_id[:2])
    cortex_dist_resampled = resmaple_dist_map(cortex_name, target_shape)
    hev_dist_resampled = resmaple_dist_map(hev_name, target_shape)

    dc_density = cell_measurements_file['{}_xcr1_within_{}'.format(file_id, distance)]
    #normalize by total numbers of cells
    # dc_density = np.array(dc_density) / cell_xyz.shape[0]

    # convert into mask coordinates
    xrc1_coords_resampled = cell_xyz / np.array([0.555, 0.555, 4]) / get_image_shape(file) * target_shape
    zyx_coords = np.flip(xrc1_coords_resampled.astype(np.int), axis=1)
    cell_cortex_dists = cortex_dist_resampled[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]
    cell_hev_dists = hev_dist_resampled[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]

    xrc1_coords_resampled = cell_xyz / np.array([0.555, 0.555, 4]) / get_image_shape(file) * np.array(dc_density.shape)
    zyx_coords = np.flip(xrc1_coords_resampled.astype(np.int), axis=1)
    dc_densities = np.array(dc_density)[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]
    return dc_densities, cell_cortex_dists, cell_hev_dists

def measure_densities(file_id, file, cell_xyz, distance=100, time_index=0, density_cell_name='xcr1'):
    cell_measurements_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/density_maps.hdf', 'r')

    dc_density = cell_measurements_file['{}_{}_within_{}_tp_{}'.format(file_id, density_cell_name.lower(), distance, time_index)]
    # normalize by total numbers of cells
    # dc_density = np.array(dc_density) / cell_xyz.shape[0]

    # convert into mask coordinates

    xrc1_coords_resampled = cell_xyz / np.array([0.555, 0.555, 4]) / get_image_shape(file) * np.array(dc_density.shape)
    zyx_coords = np.flip(xrc1_coords_resampled.astype(np.int), axis=1)
    dc_densities = np.array(dc_density)[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]
    return dc_densities

def measure_distances_to_hevs(file_id, file, cell_xyz):
    """
    Sample from DC density and distance maps of HEVs and cortex for a given cell pipulation
    :param file_id:
    :param cell_xyz:
    :return:
    """

    target_shape = np.array([900, 900, 900])
    # hev_name = '{}_hevs'.format(file_id)
    hev_name = '{}'.format(file_id)
    hev_dist_resampled = resmaple_dist_map(hev_name, target_shape)

    # convert into mask coordinates
    xrc1_coords_resampled = cell_xyz / np.array([0.555, 0.555, 4]) / get_image_shape(file) * target_shape
    zyx_coords = np.flip(xrc1_coords_resampled.astype(np.int), axis=1)
    cell_hev_dists = hev_dist_resampled[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]

    return cell_hev_dists

def cell_motility_vs_cell_density(imaris_file, track_id, density_id, elapsed_time_s, num_bootstraps, min_track_duration,
                         cell_name, density_cell_name='xcr1', distance_threshold=100, n_query=10, alpha=0.7,
                         weight_fn='tricubic', window_sigma=0.5, raw_data=False):

    cell_measurements_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/density_maps.hdf', 'r')

    # 	Deep T, IFR - Define by proximity to both T cells and cortex, B cell
    cell_xyz, time_indices, tracks = get_spots(imaris_file, cell_name)

    density_name = '{}_{}_within_{}'.format(density_id, density_cell_name.lower(), distance_threshold)
    densities = [cell_measurements_file[density] for density in list(cell_measurements_file) if density_name in density]
    if len(densities) > 1:
        density = np.mean(np.stack(densities, axis=0), axis=0)
    else:
        density = densities[0]

    ### Assign cells to different regions
    # convert into mask coordinates
    coords_resampled = cell_xyz / np.array([0.555, 0.555, 4]) / get_image_shape(imaris_file) * np.array(density.shape)
    zyx_coords = np.flip(coords_resampled.astype(np.int), axis=1)

    density_at_cells = np.array(density)[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]
    #visualize distance map
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(np.array(dc_density))


    filtered_tracks = filter_tracks(imaris_file, track_id, cell_xyz, time_indices, tracks, elapsed_time_s,
                                             min_track_duration, split_by_subregion=False)


    _, motility = compute_displacements(cell_xyz, filtered_tracks, time_indices, elapsed_time_s)

    x = density_at_cells[[track[0] for track in filtered_tracks]]
    if raw_data:
        plt.plot(x, motility, '.')
    else:
        bootstrap_and_plot(x, motility, num_bootstraps=num_bootstraps, alpha=alpha, zero_bias=True,
                           n_query_points=n_query, weight_fn=weight_fn, window_sigma=window_sigma)

    plt.xlabel('# of {} within {} um'.format(density_cell_name, distance_threshold))
    plt.ylabel('Motility coefficient')
    # plt.savefig('figures/DC_motility_by_dc_density_{}_region_split_{}.pdf'.format(file_id, split_by_subregion))

def compute_t_cell_zone_mask(image_shape, all_t_coords, num_samples=150, percentile=0.5, threshold=130, save_name='25_t_cell_mask'):
    image_size_um = np.flip(np.array(image_shape)) * np.array([4, 0.555, 0.555])

    zz, yy, xx = np.meshgrid(np.linspace(0, image_size_um[0], num_samples),
                             np.linspace(0, image_size_um[1], num_samples),
                             np.linspace(0, image_size_um[2], num_samples), indexing='ij')

    all_coords = np.stack([np.ravel(xx), np.ravel(yy), np.ravel(zz)], axis=1)
    #Compute the nth percentile of closesest T cells to each pointin space
    batch_size = 5000
    #iterate through batches to save memory
    def process_batch(batch_index):
        print('{} of {}'.format(batch_index, all_coords.shape[0] // batch_size))
        dists = np.linalg.norm(all_t_coords[:, None, :] - all_coords[batch_index * batch_size:
                                    (batch_index + 1) * batch_size], axis=2)
        return np.percentile(dists, percentile, axis=0)
    pool = ThreadPool(8)
    counts = pool.map(process_batch, range(all_coords.shape[0] // batch_size + 1))
    dist_prctile_image = np.reshape(np.concatenate(counts, axis=0), 3*[num_samples])


    # hev_cortex_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/with_surface_masks/distance_transforms.h5', mode='r')
    # cortex_dist = hev_cortex_file['48_cortex']
    # cortex_dist_resampled = ndi.zoom(cortex_dist, np.array(3 * [num_samples]) / np.array(cortex_dist.shape))

    # scaled_points = num_samples * np.flip(all_t_coords, axis=1) / (np.array([4, 0.555, 0.555]) * np.array(image_shape))
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(dist_prctile_image)
    #     # viewer.add_image(dist_prctile_image < 130)
    #     # viewer.add_image(dist_prctile_image < 100)
    #     viewer.add_points(scaled_points, size=3, n_dimensional=True)
    # #     viewer.add_image(cortex_dist_resampled, blending='additive')

    #Threshold the percentile image to defin a zone for the T cell mask
    t_cell_mask = dist_prctile_image < threshold
    with h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/density_maps.hdf', 'r+') as out_file:
        if save_name in out_file:
            del out_file[save_name]
        out_file.create_dataset(name=save_name, data=t_cell_mask)

def compute_cell_density_gradient(distance_threshold=100, file_id=25, num_samples=150, cell_name='XCR1'):
    """
    Compute number of DCs within threshold um at each pixel in resampled volume of the image
    """
    root_path = '/Users/henrypinkard/Desktop/imaris_analysis/'
    filepath = root_path + '{}.ims'.format(file_id)
    with h5py.File(filepath, mode='r') as file:
        cell_coords, time_indices, tracks = get_spots(file, cell_name)
        coords_at_0 = cell_coords[time_indices == 0]
        image_shape = np.flip(get_image_shape(file))
    image_size_um = np.array(image_shape) * np.array([4, 0.555, 0.555])
    print('Computing gradients {}'.format(file_id))

    zz, yy, xx = np.meshgrid(np.linspace(0, image_size_um[0], num_samples),
                             np.linspace(0, image_size_um[1], num_samples),
                             np.linspace(0, image_size_um[2], num_samples), indexing='ij')
    all_coords = np.stack([np.ravel(xx), np.ravel(yy), np.ravel(zz)], axis=1)
    batch_size = 5000
    #iterate through batches to save memory
    def process_batch(batch_index):
        print('{} of {}'.format(batch_index, all_coords.shape[0] // batch_size))
        batch_coords = all_coords[batch_index * batch_size:(batch_index + 1) * batch_size]
        distances_batch = np.linalg.norm(coords_at_0[:, None, :] - batch_coords, axis=2)

        mask = distances_batch < distance_threshold
        batch = np.repeat(coords_at_0[:, :, None], batch_coords.shape[0], 2)
        batch_relative_coords = batch - batch_coords.T
        batch_local_relative_coords = batch_relative_coords * mask[:, None, :]
        return np.mean(batch_local_relative_coords, axis=0)

    # for batch_index in range(600, all_coords.shape[0] // batch_size + 1):
    #     process_batch(batch_index)
    pool = ThreadPool(8)
    vecs = pool.map(process_batch, range(all_coords.shape[0] // batch_size + 1))
    all_vec = np.concatenate(vecs, axis=1)

    count_image = np.reshape(all_vec, [3, num_samples, num_samples, num_samples])
    with h5py.File(root_path + 'density_maps.hdf', 'a') as out_file:
        out_file.create_dataset(name='{}_{}_gradient_{}'.format(file_id, cell_name.lower(), distance_threshold), data=count_image)

def compute_density_gradient_motility_angles(imaris_file, density_file_id, elapsed_time_s,
                 cell_name,  motility_cutoff=0.04, density_cell_name='xcr1', distance_threshold=100):

    cell_measurements_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/density_maps.hdf', 'r')
    cell_xyz, time_indices, tracks = get_spots(imaris_file, cell_name)
    gradient = cell_measurements_file['{}_{}_gradient_{}'.format(density_file_id, density_cell_name.lower(), distance_threshold)]

    ### Assign cells to different regions
    # convert into mask coordinates
    coords_resampled = cell_xyz / np.array([0.555, 0.555, 4]) / get_image_shape(imaris_file) * np.array(gradient.shape[1:])
    zyx_coords = np.flip(coords_resampled.astype(np.int), axis=1)

    gradient_in_mem = np.array(gradient)
    gradient_at_cells = np.stack([gradient_in_mem[i][(zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2])] for i in range(3)], axis=0)
    #gradients are in xyz format


    _, motility = compute_displacements(cell_xyz, tracks, time_indices, elapsed_time_s)

    cell_xyzs_initial = np.stack([cell_xyz[track[0]] for i, track in enumerate(tracks) ], axis=0)
    cell_xyzs_final = np.stack( [cell_xyz[track[-1]] for i, track in enumerate(tracks) ], axis=0)
    movement_deltas = cell_xyzs_final - cell_xyzs_initial
    gradients = np.stack([gradient_at_cells[:, track[0]] for i, track in enumerate(tracks)], axis=0)
    gradient_magnitudes = np.linalg.norm(gradients, axis=1)

    angles = np.arccos(np.sum(movement_deltas * gradients, axis=1
                     ) / (np.linalg.norm(movement_deltas, axis=1) * np.linalg.norm(movement_deltas, axis=1)))

    valid_mask = np.logical_and(motility > np.std(motility), gradient_magnitudes > np.std(gradient_magnitudes))

    return angles[valid_mask]


def compute_xcr1_density_time_series_for_all_spot_times(file, id, name, distance=100):
    coords, time_indices, tracks = get_spots(file, name)
    tp_denisities = []

    xcr1_xyz, xcr1_times, _ = get_spots(file, 'XCR1')
    for t_index in set(time_indices):
        density = measure_densities(id, file, coords, distance, time_index=t_index)
        tp_denisities.append(density / coords.shape[0])
    return tp_denisities

def compute_xcr1_density_time_series(file, id, name, elapsed_time, distance=100):
    coords, time_indices, tracks = get_spots(file, name)
    tp_denisities = []
    tp_time_indices = []

    xcr1_xyz, xcr1_times, _ = get_spots(file, 'XCR1')
    for t_index in set(time_indices):
        coords_at_tp = coords[time_indices == t_index]
        tp_time_indices.append(coords_at_tp.shape[0] * [elapsed_time[t_index] / 60])

        density_control = measure_densities(id, file, coords_at_tp, distance, time_index=t_index)
        tp_denisities.append(density_control / np.sum(xcr1_times == t_index))
    return np.concatenate(tp_time_indices), np.concatenate(tp_denisities)

def compute_xcr1_density_at_time_index(file, id, name, t_index, distance=100):
    coords, time_indices, tracks = get_spots(file, name)
    coords_at_tp = coords[time_indices == t_index]

    density_control = measure_densities(id, file, coords_at_tp, distance, time_index=t_index)
    xcr1_xyz, xcr1_times, _ = get_spots(file, 'XCR1')
    return density_control / np.sum(xcr1_times == t_index)

    # N = 20
    # theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    # radii = 10 * np.random.rand(N)
    # width = np.pi / 4 * np.random.rand(N)
    #
    # ax = plt.subplot(111, projection='polar')
    # bars = ax.bar(theta, radii, width=width, bottom=0.0)

# if __name__ ==
#
# time_cal_filepath4849 = '/Users/henrypinkard/Desktop/imaris_analysis/4849.ims'
# elapsed_time_s4849 = read_time_calibration(time_cal_filepath4849)
# file4849 = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/4849_cropped_to_48.ims', mode='r')
#
# compute_density_gradient_motility_angles(file4849, '4849_cropped_to_48', elapsed_time_s4849, 'XCR1')
#
#
# time_cal_filepath25 = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/25_uncorrected.ims'
# elapsed_time_s25 = read_time_calibration(time_cal_filepath25)
# file25 = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/25.ims', mode='r')
#
# compute_density_gradient_motility_angles(file25, '25', elapsed_time_s25, 'XCR1')
#
#
# plt.show()
# pass



# time_cal_filepath45 = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/45_uncorrected.ims'
# file45 = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/45.ims', 'r')
#
# elapsed_time_s45 = read_time_calibration(time_cal_filepath45)
# min_track_duration = 10 * 60
#
# plt.figure()
# cell_motility_vs_cell_density(file45, None, '45', elapsed_time_s45,
#                           num_bootstraps=5, min_track_duration=min_track_duration,
#                           density_cell_name='XCR1', cell_name='GFP', raw_data=False, alpha=0.5)

#
# time_cal_filepath25 = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/25_uncorrected.ims'
# file25 = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/25.ims', 'r')
#
# time_cal_filepath21 = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/21_uncorrected.ims'
# file21 = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/21.ims', 'r')

pass