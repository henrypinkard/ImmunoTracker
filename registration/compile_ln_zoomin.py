import numpy as np
import os
from stitcher import stitch_single_channel, apply_intra_stack_registration
from stitcher import compute_inter_stack_registrations
from utility import x_corr_register_3D, anisotropic_x_corr_register_3D
from imaris_writing import stitch_register_imaris_write
from data_reading import open_magellan, read_raw_data
from optimization_stitcher import optimize_timepoint_stacks, optimize_inter_stack_stitching, optimize_z_over_time
from dual_logging import DualLogger
import sys
from pathlib import Path
import csv
from imariswriter import ImarisJavaWrapper



def estimate_background(p_zyxc_stacks, nonempty_pixels):
    """
    Estiamte a background pixel value for every channel in raw_stacks
    :param raw_stack:
    :return:
    """
    print('Computing background')
    all_pix = {}
    for position_index in p_zyxc_stacks.keys():
        for channel_index in range(p_zyxc_stacks[position_index].shape[-1]):
            if channel_index not in all_pix:
                all_pix[channel_index] = np.ravel(
                    p_zyxc_stacks[position_index][..., channel_index][nonempty_pixels[position_index]])
            else:
                all_pix[channel_index] = np.concatenate((all_pix[channel_index],
                                                         np.ravel(p_zyxc_stacks[position_index][..., channel_index][
                                                                      nonempty_pixels[position_index]])))
        if all_pix[0].size > 1e8:
            break  # dont need every last pixel
    all_pix = np.stack(list(all_pix.values()))
    backgrounds = []
    for channel_pix in all_pix:
        backgrounds.append(np.mean(channel_pix[channel_pix <= np.percentile(channel_pix, 25)]))
    return np.array(backgrounds)


def convert_single_tp(magellan, metadata, ID, param_cache_dir, time_index):
    input_filter_sigma = 2
    reverse_rank_filter=True

    # load time reg params if possible
    t_zyx_global_shifts = np.zeros((1, 3), np.int)
    t_p_zyx_residual_shifts = np.zeros((1, metadata['num_positions'], 3), dtype=np.int)
    saved_name = '{}{}_xcorr_params.npz'.format(param_cache_dir, ID + '_time_reg')
    if os.path.isfile(saved_name):
        with np.load(saved_name) as loaded:
            if 't_zyx_global_shifts' in loaded and 't_p_zyx_residual_shifts' in loaded:
                print('Loaded params from: ' + saved_name)
                t_zyx_global_shifts = loaded['t_zyx_global_shifts']  # so stitcher can use it
                t_p_zyx_residual_shifts = loaded['t_p_zyx_residual_shifts']
    # load stack params if possible
    param_cache_name = ID + '_tp{}'.format(time_index)
    saved_name = '{}{}_optimized_params.npz'.format(param_cache_dir, param_cache_name)
    if os.path.isfile(saved_name) and ID:
        with np.load(saved_name) as loaded:
            if 'p_yx_translations' in loaded:
                print('Loaded params from: ' + saved_name)
                p_yx_translations = loaded['p_yx_translations']

    param_cache_name = ID
    saved_name = '{}{}_optimized_stitch_params.npz'.format(param_cache_dir, param_cache_name)
    if os.path.isfile(saved_name):
        with np.load(saved_name) as loaded:
            if 'p_zyx_stitch' in loaded:
                print('Loaded params from: ' + saved_name)
                p_zyx_stitch = loaded['p_zyx_stitch']


    p_zyx_stitch[:, 1:] = -p_zyx_stitch[:, 1:]
    t_p_zyx_residual_shifts = -np.copy(t_p_zyx_residual_shifts)


    # merge stitching zyx translations and the ones derived from timepoint cross correlations
    t_p_zyx_translations = np.round(t_p_zyx_residual_shifts + p_zyx_stitch).astype(np.int)


    # make all global shifts nonnegative ints
    t_zyx_global_shifts -= np.min(t_zyx_global_shifts, axis=0)

    # make sure again their median is centered at 0
    median_z_shift = np.median(np.reshape(t_p_zyx_translations, [-1, 3])[:, 0]).astype(np.int)
    t_p_zyx_translations[:, :, 0] -= median_z_shift
    median_z_shift_0_based = -np.min(np.reshape(-t_p_zyx_translations, [-1, 3])[:, 0]).astype(np.int)

    z_shift_range = np.ptp(np.reshape(t_p_zyx_translations, [-1, 3])[:, 0])
    # compute the size of teh stiched image accounting for movements in z
    stitched_image_size = [z_shift_range + metadata['max_z_index'] - metadata['min_z_index'] + 1,
                           (1 + np.ptp(metadata['row_col_coords'][:, 0], axis=0)) * (
                                   metadata['tile_shape'][0] - metadata['tile_overlaps'][0]),
                           (1 + np.ptp(metadata['row_col_coords'][:, 1], axis=0)) * (
                                   metadata['tile_shape'][1] - metadata['tile_overlaps'][1])]


    # add in time point to timepoint registrations for the final imaris size
    imaris_size = np.array(stitched_image_size) + np.max(t_zyx_global_shifts, axis=0).astype(np.int)

    num_channels = metadata['num_channels']
    byte_depth = metadata['byte_depth']

    raw_stacks, nonempty_pixels, timestamp = read_raw_data(magellan, metadata, time_index=time_index,
                                                           reverse_rank_filter=reverse_rank_filter,
                                                           input_filter_sigma=input_filter_sigma)
    all_channels = []
    for channel_index in range(num_channels):
        top_stack_relative_to_median = np.min(-t_p_zyx_translations[time_index, :, 0]).astype(np.int)
        stitched = stitch_single_channel(raw_stacks, p_zyx_translations=t_p_zyx_translations[time_index],
                                         p_yx_translations=p_yx_translations,
                                         tile_overlap=metadata['tile_overlaps'],
                                         row_col_coords=metadata['row_col_coords'], channel_index=channel_index)

        tp_registered = np.zeros(imaris_size.astype(np.int), dtype=np.uint8 if int(byte_depth) == 1 else np.uint16)
        tp_registered[median_z_shift_0_based + top_stack_relative_to_median + t_zyx_global_shifts[time_index, 0]:
                      median_z_shift_0_based + top_stack_relative_to_median + t_zyx_global_shifts[time_index, 0] +
                      stitched.shape[0], t_zyx_global_shifts[time_index, 1]:t_zyx_global_shifts[time_index, 1] +
                                    stitched.shape[1], t_zyx_global_shifts[time_index, 2]:t_zyx_global_shifts[time_index, 2] + stitched.shape[2]] = stitched
        all_channels.append(tp_registered)
    return np.stack(all_channels, axis=3)



home = str(Path.home())
# magellan_dir, ID, param_cache_dir, time_index,
storage_path = home + '/lymphosight_data/'

imaris_dir = storage_path + 'imaris_files'
raw_data_dir = storage_path + 'raw_data/'
param_cache_dir = storage_path + 'optimized_params' + '/'
data_list = home + '/GitRepos/LymphoSight/LymphosightDatasets.csv'

with open(data_list, 'r') as f:
    reader = csv.DictReader(f)
    experiments = list(reader)
    column_names = reader.fieldnames

def get_dataset_path(ID):
    ex = [e for e in experiments if e['ID'] == ID][0]
    path = '{} {}{}{}'.format(ex['Date'], ex['Folder'], os.sep, ex['Experiment'])
    return path



#TODO: Load time point from each dataset
data_path = get_dataset_path('48')
magellan_dir = raw_data_dir + data_path
magellan48, metadata48 = open_magellan(magellan_dir)
volume48 = convert_single_tp(magellan48, metadata48, '48', param_cache_dir, 0)

data_path = get_dataset_path('49')
magellan_dir = raw_data_dir + data_path
magellan49, metadata49 = open_magellan(magellan_dir)
volume49 = convert_single_tp(magellan49, metadata49, '49', param_cache_dir, 0)

max_shape = np.maximum(np.array(volume48.shape[:3]), np.array(volume49.shape[:3]))
v48_padded = np.zeros(max_shape, dtype=volume48.dtype)
v49_padded = np.zeros(max_shape, dtype=volume48.dtype)

v48_padded[:volume48.shape[0], :volume48.shape[1], :volume48.shape[2]] = volume48[..., 0]
v49_padded[:volume49.shape[0], :volume49.shape[1], :volume49.shape[2]] = volume49[..., 0]

offsets = x_corr_register_3D(v48_padded, v49_padded, np.array(v48_padded.shape) // 2) #[-10 692 857]

offset48 = np.copy(offsets)
offset49 = np.copy(offsets)
offset48[offset48 > 0] = 0
offset48 = -offset48
offset49[offset49 < 0] = 0

imaris_size_x = max_shape[2]
imaris_size_y = max_shape[1]
imaris_size_z = max_shape[0]
num_frames = 20
byte_depth = 1

with ImarisJavaWrapper(imaris_dir, '48-49_fusion_zoomin', (imaris_size_x, imaris_size_y, imaris_size_z), byte_depth, 6,
                       num_frames, magellan49.pixel_size_xy_um, float(magellan49.pixel_size_z_um)) as writer:
    for time_index in range(20):
        data = np.zeros([max_shape[0], max_shape[1], max_shape[2], 6], dtype=volume48.dtype)
        if time_index == 0:
            data[offset48[0]:offset48[0] + volume48.shape[0],
                         offset48[1]:offset48[1] + volume48.shape[1],
                        offset48[2]:offset48[2] + volume48.shape[2]] = volume48
        else:
            data[offset49[0]:offset49[0] + volume49.shape[0],
                    offset49[1]:offset49[1] + volume49.shape[1],
                     offset49[2]:offset49[2] + volume49.shape[2]] = convert_single_tp(
                            magellan49, metadata49, '49', param_cache_dir, time_index - 1)
        elapsed_time_ms = 0
        for channel_index in range(6):
            stack = data[..., channel_index]
            print('Frame: {} of {}, Channel: {}'.format(
                time_index + 1, num_frames, channel_index + 1))
            for z_index, image in enumerate(stack):
                image = image.astype(np.uint8 if byte_depth == 1 else np.uint16)
                # add image to imaris writer
                writer.write_z_slice(image, z_index, channel_index, time_index, elapsed_time_ms)
print('Finshed!')