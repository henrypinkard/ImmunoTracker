import numpy as np
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
from stitcher import stitch_single_channel
from stitcher import compute_inter_stack_registrations
from utility import x_corr_register_3D
from imaris_writing import stitch_register_imaris_write
from data_reading import open_magellan, read_raw_data
from optimization_stitcher import optimize_timepoint

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

def convert(magellan_dir, position_registrations=None, register_timepoints=True, save_memory=False, input_filter_sigma=None,
            output_dir=None, output_basename=None, intra_stack_registration_channels=[1, 2, 3, 4, 5],
            inter_stack_registration_channels=[0], num_time_points=None, inter_stack_max_z=15, timepoint_registration_channel=0, n_cores=8,
            reverse_rank_filter=False, optimization_log_dir='.'):
    """
    Convert Magellan dataset to imaris, stitching tiles together and performing registration corrections as specified
    :param magellan_dir: directory of magellan data to be converted
    :param position_registrations: what algorithms to use. None: stitch and make Imaris file with no corrections.
    : 'optimize': run intravital correction optimization. 'fast_register': use cross correlation and least squares to
    : register different stacks
    :param: register_timepoints: whether to register stitched volumes to one anothrer form tp to tp
    :param save_memory: use memory mapping to keep stuff on disk when possible
    :param input_filter_sigma: apply gaussian filter to each 2D slice of raw data before doing anyhting with it
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
    if num_time_points is None:
        num_time_points=metadata['num_frames']
    for frame_index in range(num_time_points):
        translation_params = np.zeros((metadata['num_positions'], 3), dtype=np.int)
        registration_params = metadata['num_positions'] * [np.zeros((metadata['max_z_index']
                                                                     - metadata['min_z_index'] + 1, 2))]

        raw_stacks, nonempty_pixels, timestamp = read_raw_data(magellan, metadata, time_index=frame_index,
                    reverse_rank_filter=reverse_rank_filter, input_filter_sigma=input_filter_sigma, save_ram=save_memory)
        if backgrounds is None:
            # get backgrounds from first time point
            backgrounds = estimate_background(raw_stacks, nonempty_pixels)
        if position_registrations is not None:
            if position_registrations == 'optimize':
                registration_params, translation_params = optimize_timepoint(raw_stacks, nonempty_pixels,
                               metadata['row_col_coords'], metadata['tile_overlaps'],
                               intra_stack_channels=intra_stack_registration_channels,
                               inter_stack_channels=inter_stack_registration_channels,
                                optimization_log_dir=optimization_log_dir,
                                                            name=output_basename + '_tp{}'.format(frame_index))
            elif position_registrations == 'fast_register':
                translation_params = compute_inter_stack_registrations(raw_stacks, nonempty_pixels, registration_params,
                                metadata, max_shift_z=inter_stack_max_z, channel_indices=inter_stack_registration_channels,
                                                                       backgrounds=backgrounds, n_cores=n_cores)

        # Update the size of stitched image based on XYZ translations
        if stitched_image_size is None:
            stitched_image_size = [np.ptp(np.round(translation_params[:, 0])) + metadata['max_z_index'] - metadata['min_z_index'] + 1,
                   (1 + np.ptp(metadata['row_col_coords'][:, 0], axis=0)) * (metadata['tile_shape'][0] - metadata['tile_overlaps'][0]),
                   (1 + np.ptp(metadata['row_col_coords'][:, 1], axis=0)) * (metadata['tile_shape'][1] - metadata['tile_overlaps'][1])]
        else:
            #expand stitched image size if stack registrations have made it bigger at this TP
            stitched_image_size[0] = max(stitched_image_size[0], np.ptp(translation_params[:, 0]) + metadata['max_z_index'] - metadata['min_z_index'] + 1)

        #Register 3D volumes of successive timepoints to one another
        #create a stitched version for doing timepoint to timepoint registrations
        timepoint_registration = np.zeros(3)
        if metadata['num_frames'] > 1 and register_timepoints:
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
                timepoint_registration = x_corr_register_3D(previous_stitched, stitched,
                                                             max_shift=np.array(stitched.shape) // 2)
            previous_stitched = stitched
        all_params.append((registration_params, translation_params, timepoint_registration))

    registration_series = np.stack([p[0] for p in all_params])
    translation_series = np.stack([p[1] for p in all_params])
    timepoint_registrations = np.stack([p[2] for p in all_params])
    #take cumulitive shift
    abs_timepoint_registrations = np.cumsum(timepoint_registrations, axis=0).astype(np.int)
    #make all positive
    abs_timepoint_registrations -= np.min(abs_timepoint_registrations, axis=0)
    #add in extra space for timepoint registrations
    imaris_size = np.array(stitched_image_size) + np.max(abs_timepoint_registrations, axis=0).astype(np.int)

    stitch_register_imaris_write(output_dir, output_basename, imaris_size, magellan, metadata, registration_series,
                                 translation_series, abs_timepoint_registrations, input_filter_sigma=input_filter_sigma,
                                 reverse_rank_filter=reverse_rank_filter, save_memory=save_memory)
