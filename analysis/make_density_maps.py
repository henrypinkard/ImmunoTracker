import numpy as np
import h5py
import scipy.ndimage as nd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from util import *
import scipy.ndimage as ndi
import napari
import scipy.ndimage as ndi
# from dask.distributed import Client
# client = Client()
from multiprocessing.pool import ThreadPool




# ######### T cell density map (25)
# filepath25 = '/Users/henrypinkard/Desktop/imaris_analysis/25.ims'
# with h5py.File(filepath25, mode='r') as file25:
#     coords_polyclonal, time_indices_polyclonal, tracks_polyclonal = get_spots(file25, 'e670')
#     coords_ot1, _, _ = get_spots(file25, 'RFP')
#     coords_ot2, _, _ = get_spots(file25, 'VPD')
#     image_shape = file25['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data'].shape
# all_t_coords = np.concatenate([coords_polyclonal, coords_ot1, coords_ot2], axis=0)
# compute_t_cell_zone_mask(image_shape, all_t_coords, save_name='25_t_cell_mask')


# ######### T cell density map (48)
# filepath4849 = '/Users/henrypinkard/Desktop/imaris_analysis/48-49_cropped_to_48.ims'
# with h5py.File(filepath4849, mode='r') as file4849:
#     coords_polyclonal, time_indices_polyclonal, tracks_polyclonal = get_spots(file4849, 'GFP')
#     #Take only TP 0 (i.e. datase 48)
#     coords_polyclonal = coords_polyclonal[time_indices_polyclonal == 0]
#     image_shape = file4849['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data'].shape
# all_t_coords = coords_polyclonal
# compute_t_cell_zone_mask(image_shape, all_t_coords, save_name='48_t_cell_mask', percentile=0.5, threshold=200)




def distance_to_static_ot1_ot2():
    num_samples=150
    filepath4849 = '/Users/henrypinkard/Desktop/imaris_analysis/48-49.ims'
    with h5py.File(filepath4849, mode='r') as file4849:
        coords_xcr1, time_indices_xcr1, tracks_xcr1 = get_spots(file4849, 'XCR1')
        coords_ot1, time_indices_ot1, tracks_ot1 = get_spots(file4849, 'RFP')
        coords_ot2, time_indices_ot2, tracks_ot2 = get_spots(file4849, 'VPD')
        image_shape = file4849['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data'].shape
    time_cal_filepath49 = '/Users/henrypinkard/Desktop/imaris_analysis/48-49.ims'
    elapsed_time_s49 = read_time_calibration(time_cal_filepath49)
    image_size_um = np.array(image_shape) * np.array([4, 0.555, 0.555])

    zz, yy, xx = np.meshgrid(np.linspace(0, image_size_um[0], num_samples),
                             np.linspace(0, image_size_um[1], num_samples),
                             np.linspace(0, image_size_um[2], num_samples), indexing='ij')

    all_coords = np.stack([np.ravel(xx), np.ravel(yy), np.ravel(zz)], axis=1)


    displacements_ot1 = []
    for track in tracks_ot1:
        if time_indices_ot1[track[-1]] - time_indices_ot1[track[0]] < 5:
            continue
        displacement_sq_over_t = np.sum((coords_ot1[track[-1]] - coords_ot1[track[0]]) ** 2) / (
                elapsed_time_s49[time_indices_ot1[track[-1]]] - elapsed_time_s49[time_indices_ot1[track[0]]])
        displacements_ot1.append((displacement_sq_over_t, track))
    displacements_ot2 = []
    for track in tracks_ot2:
        if time_indices_ot2[track[-1]] - time_indices_ot2[track[0]] < 5:
            continue
        displacement_sq_over_t = np.sum((coords_ot2[track[-1]] - coords_ot2[track[0]]) ** 2) / (
                elapsed_time_s49[time_indices_ot2[track[-1]]] - elapsed_time_s49[time_indices_ot2[track[0]]])
        displacements_ot2.append((displacement_sq_over_t, track))

    # plt.hist(displacements_ot1, 15)
    # plt.hist(displacements_ot2, 15)

    static_ot1_tracks = np.concatenate([track for disp_sq_over_t, track in displacements_ot1 if disp_sq_over_t < 0.4])
    static_ot2_tracks = np.concatenate([track for disp_sq_over_t, track in displacements_ot2 if disp_sq_over_t < 0.4])
    coords_ot1_staitc = coords_ot1[static_ot1_tracks]
    coords_ot2_staitc = coords_ot2[static_ot2_tracks]

    batch_size = 5000

    # iterate through batches to save memory
    def process_batch(t_coords, batch_index):
        print('{} of {}'.format(batch_index, all_coords.shape[0] // batch_size))
        dists = np.linalg.norm(t_coords[:, None, :] - all_coords[batch_index * batch_size:
                                                                     (batch_index + 1) * batch_size], axis=2)
        return np.min(dists, axis=0)
    pool = ThreadPool(8)

    counts = pool.map(lambda x: process_batch(coords_ot1_staitc, x), range(all_coords.shape[0] // batch_size + 1))
    min_dist_ot1 = np.reshape(np.concatenate(counts, axis=0), 3 * [num_samples])

    counts = pool.map(lambda x: process_batch(coords_ot2_staitc, x), range(all_coords.shape[0] // batch_size + 1))
    min_dist_ot2 = np.reshape(np.concatenate(counts, axis=0), 3 * [num_samples])


    with h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/density_maps.hdf') as out_file:
        out_file.create_dataset(name='48_static_ot1_distance', data=min_dist_ot1)
        out_file.create_dataset(name='48_static_ot2_distance', data=min_dist_ot2)


def compute_t_cell_density(distance_threshold=60, file_id=21, num_samples=150):
    """
    Compute number of DCs within threshold um at each pixel in resampled volume of the image
    """
    filepath = root_path + '{}.ims'.format(file_id)
    with h5py.File(filepath, mode='r') as file:
        cell_coords, time_indices, tracks = get_spots(file, 'VPD')
        image_shape = file['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data'].shape
    image_size_um = np.array(image_shape) * np.array([4, 0.555, 0.555])


    zz, yy, xx = np.meshgrid(np.linspace(0, image_size_um[0], num_samples),
                             np.linspace(0, image_size_um[1], num_samples),
                             np.linspace(0, image_size_um[2], num_samples), indexing='ij')
    all_coords = np.stack([np.ravel(xx), np.ravel(yy), np.ravel(zz)], axis=1)
    batch_size = 5000
    #iterate through batches to save memory
    def process_batch(batch_index):
        print('{} of {}'.format(batch_index, all_coords.shape[0] // batch_size))
        distances_batch = np.linalg.norm(cell_coords[:, None, :] - all_coords[batch_index * batch_size:
                                    (batch_index + 1) * batch_size], axis=2)
        return np.sum(distances_batch < distance_threshold, axis=0)
    pool = ThreadPool(8)
    counts = pool.map(process_batch, range(all_coords.shape[0] // batch_size + 1))
    all_counts = np.concatenate(counts, axis=0)

    count_image = np.reshape(all_counts, 3*[num_samples])
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(count_image)
    with h5py.File(root_path + 'density_maps.hdf', 'a') as out_file:
        out_file.create_dataset(name='{}_ot1_within_{}'.format(file_id, distance_threshold), data=count_image)

compute_t_cell_density()
