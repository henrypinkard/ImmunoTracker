from pygellan import MagellanDataset
from imariswriter import ImarisJavaWrapper
import numpy as np
from scipy.ndimage import filters
import os
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# #make text on figures look good
# SMALL_SIZE = 16
# MEDIUM_SIZE = 22
# BIGGER_SIZE = 28
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
from PIL import Image
from scipy import ndimage as ndi
from scipy import signal, optimize
from itertools import combinations
import warnings
from intravital_stack import optimize_intra_stack_registrations
from stitcher import stitch_single_channel
from stitcher import compute_inter_stack_registrations
from stitcher import x_corr_register_3D

def open_magellan(path):
    """
    open a magellan dataset on disk and read all appropriate metadata fields
    :param path: path to top level magellan folder
    :return:
    """
    magellan = MagellanDataset(path)
    metadata = {}
    if magellan.summary_metadata['PixelType'] == 'GRAY8':
        metadata['byte_depth'] = 1
    else:
        metadata['byte_depth'] = 2
    metadata['num_positions'] = magellan.get_num_xy_positions()
    min_z_index, max_z_index = magellan.get_min_max_z_index()
    metadata['min_z_index'] = min_z_index
    metadata['max_z_index'] = max_z_index
    metadata['num_channels'] = len(magellan.summary_metadata['ChNames'])
    metadata['tile_overlaps'] = np.array([magellan.summary_metadata['GridPixelOverlapY'],
                                          magellan.summary_metadata['GridPixelOverlapX']])
    metadata['tile_shape'] = np.array([magellan.summary_metadata['Height'], magellan.summary_metadata['Width']])
    metadata['pixel_size_xy_um'] = magellan.summary_metadata['PixelSize_um']
    metadata['pixel_size_z_um'] = magellan.summary_metadata['z-step_um']
    metadata['num_frames'] = magellan.get_num_frames()
    num_rows, num_cols = magellan.get_num_rows_and_cols()
    metadata['num_rows'] = num_rows
    metadata['num_cols'] = num_cols
    metadata['row_col_coords'] = np.array(magellan.row_col_tuples)
    return magellan, metadata

def read_raw_data(magellan, metadata, time_index, reverse_rank_filter=False, input_filter_sigma=None):
    """
    read raw data, store in 3D arrays for each channel at each position
    :param magellan:
    :param metadata:
    :param reverse_rank_filter:
    :return:
    """
    elapsed_time_ms = ''
    raw_stacks = {}
    nonempty_pixels = {}
    for position_index in range(metadata['num_positions']):
        raw_stacks[position_index] = {}
        nonempty_pixels[position_index] = {}
        print('Reading in frame {}, position {}'.format(time_index, position_index))
        for channel_index in range(metadata['num_channels']):
            raw_stacks[position_index][channel_index] = np.zeros((metadata['max_z_index'] -
                    metadata['min_z_index'] + 1, *metadata['tile_shape']),
                                                dtype= np.uint8 if metadata['byte_depth'] == 1 else np.uint16)
            nonempty_pixels[position_index] = (metadata['max_z_index'] - metadata['min_z_index'] + 1)*[False]
            for z_index in range(raw_stacks[position_index][channel_index].shape[0]):
                if not magellan.has_image(channel_index=channel_index, pos_index=position_index,
                                        z_index=z_index + metadata['min_z_index'], t_index=time_index):
                    continue
                image, image_metadata = magellan.read_image(channel_index=channel_index, pos_index=position_index,
                                z_index=z_index + metadata['min_z_index'], t_index=time_index, read_metadata=True)
                if reverse_rank_filter:
                    #do final step of rank fitlering
                    image = ndi.percentile_filter(image, percentile=15, size=3)
                if input_filter_sigma is not None:
                    image = filters.gaussian_filter(image.astype(np.float), input_filter_sigma)

                #add in image
                raw_stacks[position_index][channel_index][z_index] = image
                nonempty_pixels[position_index][z_index] = True
                if elapsed_time_ms == '':
                    elapsed_time_ms = image_metadata['ElapsedTime-ms']
    return raw_stacks, nonempty_pixels, elapsed_time_ms

def write_imaris(directory, name, time_series, pixel_size_xy_um, pixel_size_z_um):
    timepoint0 = time_series[0][0]
    num_channels = len(timepoint0)
    t0c0 = timepoint0[0]
    imaris_size_x = t0c0.shape[2]
    imaris_size_y = t0c0.shape[1]
    imaris_size_z = t0c0.shape[0]
    num_frames = len(time_series)
    byte_depth = 1 if t0c0.dtype == np.uint8 else 2

    with ImarisJavaWrapper(directory, name, (imaris_size_x, imaris_size_y, imaris_size_z), byte_depth, num_channels,
                           num_frames, pixel_size_xy_um, float(pixel_size_z_um)) as writer:
        for time_index, (timepoint, elapsed_time_ms) in enumerate(time_series):
            for channel_index in range(len(timepoint)):
                stack = timepoint[channel_index]
                for z_index, image in enumerate(stack):
                    image = image.astype(np.uint8 if byte_depth == 1 else np.uint16)
                    #add image to imaris writer
                    print('Frame: {} of {}, Channel: {} of {}, Slice: {} of {}'.format(
                        time_index+1, num_frames, channel_index+1, num_channels, z_index, imaris_size_z))
                    writer.write_z_slice(image, z_index, channel_index, time_index, elapsed_time_ms)
    print('Finshed!')

def estimate_background(raw_stacks, nonempty_pixels):
    """
    Estiamte a background pixel value for every channel in raw_stacks
    :param raw_stack:
    :return:
    """
    print('Computing background')
    all_pix = {}
    for position_index in range(len(raw_stacks)):
        for channel_index in range(len(raw_stacks[position_index])):
            if channel_index not in all_pix:
                all_pix[channel_index] = np.ravel(raw_stacks[position_index][channel_index][nonempty_pixels[position_index]])
            else:
                all_pix[channel_index] = np.concatenate((all_pix[channel_index],
                    np.ravel(raw_stacks[position_index][channel_index][nonempty_pixels[position_index]])))
        if all_pix[channel_index].size > 1e8:
            break #dont need every last pixel
    all_pix = np.stack(list(all_pix.values()))
    backgrounds = []
    for channel_pix in all_pix:
        backgrounds.append(np.mean(channel_pix[channel_pix <= np.percentile(channel_pix, 25)]))
    return np.array(backgrounds)

def ram_efficient_stitch_register_imaris_write(directory, name, imaris_size, magellan, metadata,
                    registration_series, translation_series, abs_timepoint_registrations, input_filter_sigma=None,
                                               reverse_rank_filter=False):
    num_channels = metadata['num_channels']
    num_frames = metadata['num_frames']
    byte_depth = metadata['byte_depth']
    print('Imaris file: {}'.format(name))
    print('Imaris directory: {}'.format(directory))
    with ImarisJavaWrapper(directory, name, (int(imaris_size[2]), int(imaris_size[1]), int(imaris_size[0])), byte_depth,
                num_channels, num_frames, metadata['pixel_size_xy_um'], float(metadata['pixel_size_z_um'])) as writer:
        for time_index in range(num_frames):
            print('Frame {}'.format(time_index))
            raw_stacks, nonempty_pixels, timestamp = read_raw_data(magellan, metadata, time_index=time_index,
                                    reverse_rank_filter=reverse_rank_filter, input_filter_sigma=input_filter_sigma)
            for channel_index in range(num_channels):
                stitched = stitch_single_channel(raw_stacks, translations=translation_series[time_index],
                        registrations=registration_series[time_index], tile_overlap=metadata['tile_overlaps'],
                        row_col_coords=metadata['row_col_coords'], channel_index=channel_index)
                #fit into the larger image to account for timepoint registrations
                tp_registered = np.zeros(imaris_size)
                tp_registered[abs_timepoint_registrations[time_index, 0]:abs_timepoint_registrations[time_index, 0] + stitched.shape[0],
                        abs_timepoint_registrations[time_index, 1]:abs_timepoint_registrations[time_index, 1] + stitched.shape[1],
                       abs_timepoint_registrations[time_index, 2]:abs_timepoint_registrations[time_index, 2] + stitched.shape[2]] = stitched
                print('writing to Imaris channel {}'.format(channel_index))
                for z_index, image in enumerate(tp_registered):
                    image = image.astype(np.uint8 if byte_depth == 1 else np.uint16)
                    # add image to imaris writer
                    # print('Frame: {} of {}, Channel: {} of {}, Slice: {} of {}'.format(
                    #     time_index + 1, num_frames, channel_index + 1, num_channels, z_index, imaris_size[0]))
                    writer.write_z_slice(image, z_index, channel_index, time_index, timestamp)
    print('Finshed!')

def convert(magellan_dir, input_filter_sigma=None, do_intra_stack=True, do_inter_stack=True, do_timepoints=True,
            output_dir=None, output_basename=None, intra_stack_registration_channels=[1, 2, 3, 4, 5],
            inter_stack_registration_channels=[0], inter_stack_max_z=15, timepoint_registration_channel=0, n_cores=8,
            reverse_rank_filter=False):
    """

    :param magellan_dir: directory of magellan data to be converted
    :param input_filter_sigma: apply gaussian filter to each 2D slice of raw data before doing anyhting with it
    :param do_intra_stack: True if within z-stack corrections for intravital should be applied
    :param do_inter_stack: True if registration to align different xy tiles to one another should be applied
    :param do_timepoints: True if 3D volumes at each time point should be registered to one another
    :param output_dir: Where Imaris file should be written (defaults to parent of magellan folder
    :param output_basename: Name of imaris file to be written (defaults to same as magellan dataset)
    :param intra_stack_registration_channels: List of channel indices (0-based) to use for correcting shaking artifacts.
     Best to use all channels that have data spanning multiple z slices
    :param intra_stack_noise_model_sigma: sigma of a gaussian noise model for the likelihood of different alignments.
     should be on the order of the noise in the accuracy of the cross correlation (2 pixels?)
    :param intra_stack_zero_center_sigma: for smoothing background noise in absolute drift. Set to something slightly
     larger than the period of the movements in units of z-pixels
    :param intra_stack_likelihood_threshold_smooth: Used for selecting which slices have good signal for smoothing.
     Probably dotn need to edit this one
    :param intra_stack_likelihood_threshold: cutoff for what defines a good signal to compute registration. Probably
      needs to be empirically tested
    :param inter_stack_registration_channel: Channel to use for registering different z stacks together
    :param inter_stack_max_z: Maximum z shift among different stacks. Set smaller to speed up computations
    :param timepoint_registration_channel: Channel to use for registering different timepoints to one another
    :param n_cores: number of CPU cores to use when parallelizing inter-stack registrations.
    :return:
    """

    #autogenerate imaris name if undefined--same as
    if output_dir is None:
        output_dir = os.sep.join(magellan_dir.split(os.sep)[:-1])  # parent directory of magellan
    if output_basename is None:
        output_basename = magellan_dir.split(os.sep)[-1]  # same name as magellan acquisition

    magellan, metadata = open_magellan(magellan_dir)
    #iterate through all time points to compute all needed stitching and registration params
    all_params = []
    previous_stitched = None
    backgrounds=None
    stitched_image_size=None
    for frame_index in range(metadata['num_frames']):
        if do_intra_stack or do_inter_stack or do_timepoints:
            raw_stacks, nonempty_pixels, timestamp = read_raw_data(magellan, metadata, time_index=frame_index,
                                    reverse_rank_filter=reverse_rank_filter, input_filter_sigma=input_filter_sigma)
            if backgrounds is None:
                #get backgrounds from first time point
                backgrounds = estimate_background(raw_stacks, nonempty_pixels)

        # Intravatal breathing artifact correcttions within stack
        if do_intra_stack:
            registration_params = optimize_intra_stack_registrations(raw_stacks, nonempty_pixels,
                            np.max(metadata['tile_overlaps']),  backgrounds=backgrounds,
                                                                     use_channels=intra_stack_registration_channels)
            # registration_params = compute_intra_stack_registrations(raw_stacks, nonempty_pixels,
            #    np.max(metadata['tile_overlaps']), backgrounds=backgrounds, use_channels=intra_stack_registration_channels,
            #      sigma_noise=intra_stack_noise_model_sigma, abs_reg_bkgd_subtract_sigma=intra_stack_zero_center_sigma,
            #                                             valid_likelihood_threshold=intra_stack_likelihood_threshold)
        else:
            registration_params = metadata['num_positions'] * [np.zeros((metadata['max_z_index'] - metadata['min_z_index'] + 1, 2))]
        # XYZ stack misalignments
        if do_inter_stack:
            translation_params = compute_inter_stack_registrations(raw_stacks, nonempty_pixels, registration_params,
                            metadata, max_shift_z=inter_stack_max_z, channel_indices=inter_stack_registration_channels,
                                                                   backgrounds=backgrounds, n_cores=n_cores)
        else:
            translation_params = np.zeros((metadata['num_positions'], 3), dtype=np.int)
        # Update the size of stitched image based on XYZ translations
        if stitched_image_size is None:
            stitched_image_size = [np.ptp(translation_params[:, 0]) + metadata['max_z_index'] - metadata['min_z_index'] + 1,
                   (1 + np.ptp(metadata['row_col_coords'][:, 0], axis=0)) * (metadata['tile_shape'][0] - metadata['tile_overlaps'][0]),
                   (1 + np.ptp(metadata['row_col_coords'][:, 1], axis=0)) * (metadata['tile_shape'][1] - metadata['tile_overlaps'][1])]
        else:
            #expand stitched image size if stack registrations have made it bigger at this TP
            stitched_image_size[0] = max(stitched_image_size[0], np.ptp(translation_params[:, 0]) + metadata['max_z_index'] - metadata['min_z_index'] + 1)

        #Register 3D volumes of successive timepoints to one another
        if do_timepoints:
            #create a stitched version for doing timepoint to timepoint registrations
            stitched = stitch_single_channel(raw_stacks, translation_params, registration_params, metadata['tile_overlaps'],
                        metadata['row_col_coords'], channel_index=timepoint_registration_channel, backgrounds=backgrounds)
            if previous_stitched is not None:
                #expand the size of the shorter one to match the bigger one
                if previous_stitched.shape[0] < stitched.shape[0]:
                    previous_stitched_padded = np.ones(stitched.shape)*backgrounds[timepoint_registration_channel]
                    previous_stitched_padded[:previous_stitched.shape[0]] = previous_stitched
                    previous_stitched = previous_stitched_padded
                elif previous_stitched.shape[0] > stitched.shape[0]:
                    stitched_padded = np.ones(previous_stitched.shape)*backgrounds[timepoint_registration_channel]
                    stitched_padded[:stitched.shape[0]] = stitched
                    stitched = stitched_padded
                print('Registering timepoints')
                timepoint_registration = x_corr_register_3D(
                                previous_stitched, stitched, max_shift=np.array([10, *(np.array(raw_stacks[0][0].shape[1:]) // 2)]) )

            else:
                timepoint_registration = np.zeros(3) #first one is 0
            previous_stitched = stitched
        else:
            timepoint_registration = np.zeros(3)
        all_params.append((registration_params, translation_params, timepoint_registration))
    registration_series = np.stack([p[0] for p in all_params])
    translation_series = np.stack([p[1] for p in all_params])
    timepoint_registrations = np.stack([p[2] for p in all_params])
    #take cumulitive shift
    abs_timepoint_registrations = np.cumsum(timepoint_registrations, axis=0).astype(np.int)
    #make all positive
    abs_timepoint_registrations -= np.min(abs_timepoint_registrations)
    #add in extra space for timepoint registrations
    imaris_size = np.array(stitched_image_size) + np.max(abs_timepoint_registrations, axis=0).astype(np.int)

    ram_efficient_stitch_register_imaris_write(output_dir, output_basename, imaris_size,
            magellan, metadata, registration_series, translation_series, abs_timepoint_registrations,
                    input_filter_sigma=input_filter_sigma, reverse_rank_filter=reverse_rank_filter)


# magellan_dir = '/Users/henrypinkard/Desktop/Lymphosight/2018-6-2 4 hours post LPS/subregion timelapse_1'
# convert(magellan_dir, do_intra_stack=True, do_inter_stack=True, inter_stack_registration_channels=[5],
#                     timepoint_registration_channel=5, n_cores=8, reverse_rank_filter=True, input_filter_sigma=2)