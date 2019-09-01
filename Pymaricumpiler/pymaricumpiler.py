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
from stitcher import stitch_single_channel, apply_intra_stack_registration
from stitcher import compute_inter_stack_registrations
from utility import x_corr_register_3D, anisotropic_x_corr_register_3D
from imaris_writing import stitch_register_imaris_write
from data_reading import open_magellan, read_raw_data
from optimization_stitcher import optimize_timepoint_stacks, optimize_timepoint_stitching
from dual_logging import DualLogger
import sys

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
                all_pix[channel_index] = np.ravel(p_zyxc_stacks[position_index][..., channel_index][nonempty_pixels[position_index]])
            else:
                all_pix[channel_index] = np.concatenate((all_pix[channel_index],
                    np.ravel(p_zyxc_stacks[position_index][..., channel_index][nonempty_pixels[position_index]])))
        if all_pix[0].size > 1e8:
            break #dont need every last pixel
    all_pix = np.stack(list(all_pix.values()))
    backgrounds = []
    for channel_pix in all_pix:
        backgrounds.append(np.mean(channel_pix[channel_pix <= np.percentile(channel_pix, 25)]))
    return np.array(backgrounds)



# from PIL import Image
#
# def exporttiffstack(datacube, path, name='export'):
#     '''
#     Save 3D numpy array as a TIFF stack
#     :param datacube:
#     '''
#     if len(datacube.shape) == 2:
#         imlist = [Image.fromarray(datacube)]
#     else:
#         imlist = []
#         for i in range(datacube.shape[0]):
#             imlist.append(Image.fromarray(datacube[i,...]))
#     path = "{}{}.tif".format(path, name)
#     imlist[0].save(path, compression="tiff_deflate", save_all=True, append_images=imlist[1:])



def convert(magellan_dir, position_registrations=None, register_timepoints=True, input_filter_sigma=None,
            output_dir=None, output_basename=None, max_tp=None, min_tp=None,
            intra_stack_registration_channels=[1, 2, 3, 4, 5], stack_learning_rate=15, stack_reg=0,
            af_register_channels=[2, 5], other_register_channels=[0, 5], z_register_channels = [0],
            stitch_method='optimize',
            stitch_regularization_xy=0, stitch_regularization_z=0, stitch_downsample_factor_xy=3,
            param_cache_dir='./', log_dir='./',
            reverse_rank_filter=False, suffix='',
            stitch=True, stack=True, time_reg=True, export=True, load_params=True):
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
    :return:
    """


    #autogenerate imaris name if undefined--same as
    if output_dir is None:
        output_dir = os.sep.join(magellan_dir.split(os.sep)[:-1])  # parent directory of magellan
    if output_basename is None:
        output_basename = magellan_dir.split(os.sep)[-1] # same name as magellan acquisition
    output_basename = str(output_basename) #since it might be an ID number

    #log to both the terminal and a file
    sys.stdout = DualLogger(log_dir + output_basename + suffix + '.txt')

    magellan, metadata = open_magellan(magellan_dir)
    #iterate through all time points to compute all needed stitching and registration params
    backgrounds = None
    p_yx_series = []
    p_zyx_series = []
    last_reg_stacks = {}

    #initilize to 0 and load from file if possible
    t_zyx_global_shifts = np.zeros((max_tp - min_tp, 3), np.int)
    t_p_zyx_residual_shifts = np.zeros((max_tp - min_tp, metadata['num_positions'], 3), dtype=np.int)
    # load time reg params if possible
    saved_name = '{}{}_optimized_params.npz'.format(param_cache_dir, output_basename + '_time_reg')
    if os.path.isfile(saved_name) and load_params:
        with np.load(saved_name) as loaded:
            if 't_zyx_global_shifts' in loaded and 't_p_zyx_residual_shifts' in loaded:
                print('Loaded params from: ' + saved_name)
                t_zyx_global_shifts = loaded['t_zyx_global_shifts']  # so stitcher can use it
                t_p_zyx_residual_shifts = loaded['t_p_zyx_residual_shifts']
    #load stack params if possible
    p_yx_list = []
    for frame_index in range(min_tp, max_tp):
        # initialize to a default of 0
        p_yx_translations = metadata['num_positions'] * [np.zeros((metadata['max_z_index']
                                                                   - metadata['min_z_index'] + 1, 2))]
        # load parameters from saved file if preset
        param_cache_name = output_basename + '_tp{}'.format(frame_index)
        saved_name = '{}{}_optimized_params.npz'.format(param_cache_dir, param_cache_name)
        if os.path.isfile(saved_name) and load_params:
            with np.load(saved_name) as loaded:
                if 'p_yx_translations' in loaded:
                    print('Loaded params from: ' + saved_name)
                    p_yx_translations = loaded['p_yx_translations']
        p_yx_list.append(p_yx_translations)
    t_p_yx_translations = np.stack(p_yx_list)

    if stack or time_reg:
        pos_shift_list = []
        ##### stack + timepoint registration loop ######
        for frame_index in range(min_tp, max_tp):

                p_zyxc_stacks, nonempty_pixels, timestamp = read_raw_data(magellan, metadata, time_index=frame_index,
                            reverse_rank_filter=reverse_rank_filter, input_filter_sigma=input_filter_sigma)
                if backgrounds is None:
                    # get backgrounds from first time point
                    backgrounds = estimate_background(p_zyxc_stacks, nonempty_pixels)

                if stack: #optimize stacks
                    p_yx_translations = optimize_timepoint_stacks(p_zyxc_stacks, nonempty_pixels,
                                               intra_stack_channels=intra_stack_registration_channels,
                                               stack_learning_rate=stack_learning_rate, stack_reg=stack_reg, backgrounds=backgrounds)
                    np.savez('{}{}_optimized_params'.format(param_cache_dir, param_cache_name),
                             **{'p_yx_translations': p_yx_translations})
                    p_yx_series.append(p_yx_translations)

                if time_reg: #use temporal information to register from timepoint to timepoint, and to initialize stitching
                    #apply yx_translations and save the result on registration channels
                    pos_shift_list.append({})
                    print('applying yx_translations and registering stacks')
                    for pos_index in p_zyxc_stacks:
                        zyxc_stack = p_zyxc_stacks[pos_index]
                        if np.any(nonempty_pixels[pos_index]):
                            # reg_stack = np.min(np.stack([apply_intra_stack_registration(zyxc_stack[..., c], p_yx_translations[pos_index],
                            #                        background=np.mean(backgrounds), mode='float')
                            #                        for c in af_register_channels], axis=3), axis=3)
                            # if pos_index not in last_reg_stacks:
                            #     pos_shift_list[-1][pos_index] = np.array([0, 0, 0])  # init with shift of 0p
                            #     last_reg_stacks[pos_index] = reg_stack
                            # else:
                            #     shifts = x_corr_register_3D(last_reg_stacks[pos_index], reg_stack, np.array(reg_stack.shape) // 2)
                            #     pos_shift_list[-1][pos_index] = shifts
                            #     last_reg_stacks[pos_index] = reg_stack

                            reg_stack_xy = np.min(np.stack(
                                [apply_intra_stack_registration(zyxc_stack[..., c], p_yx_translations[pos_index],
                                                              background=np.mean(backgrounds), mode='float')
                                 for c in af_register_channels], axis=3), axis=3)
                            reg_stack_z = np.min(np.stack(
                                [apply_intra_stack_registration(zyxc_stack[..., c], p_yx_translations[pos_index],
                                                                background=np.mean(backgrounds), mode='float')
                                 for c in z_register_channels], axis=3), axis=3)
                            if pos_index not in last_reg_stacks:
                                last_reg_stacks[pos_index] = (reg_stack_xy, reg_stack_z)
                                pos_shift_list[-1][pos_index] = np.array([0, 0, 0])  # init with shift of 0p
                            else:
                                shifts = anisotropic_x_corr_register_3D(last_reg_stacks[pos_index][0], reg_stack_xy,
                                                                        last_reg_stacks[pos_index][1], reg_stack_z)
                                pos_shift_list[-1][pos_index] = shifts
                                last_reg_stacks[pos_index] = (reg_stack_xy, reg_stack_z)

        if stack:
            t_p_yx_translations = np.round(np.stack(p_yx_series)).astype(np.int)

    if time_reg:
        #compile fourier translations into useful form
        t_p_zyx_fourier_translations = np.zeros((max_tp - min_tp, len(p_zyxc_stacks.keys()), 3))
        for tp, pos_shifts in enumerate(pos_shift_list):
            for pos_index in p_zyxc_stacks.keys(): #iterate through all positions
                t_p_zyx_fourier_translations[tp, pos_index] = pos_shifts[pos_index]
        #compute timepoint relative shifts to cumulative shifts
        t_p_zyx_fourier_translations = np.cumsum(t_p_zyx_fourier_translations, axis=0)

        #Make all shifts relative to a timepoint in the middle? because the timelapses seem to stabilize by then
        t_p_zyx_fourier_translations = t_p_zyx_fourier_translations - t_p_zyx_fourier_translations[t_p_zyx_fourier_translations.shape[0] // 2, :, :]
        #negate so it works properly with stitching code
        t_p_zyx_fourier_translations[:, :, 0] = -t_p_zyx_fourier_translations[:, :, 0]
        #take median shift at each timepoint as the global shift
        t_zyx_global_shifts = np.round(np.mean(t_p_zyx_fourier_translations, axis=1)).astype(np.int)
        #include the residual as a starting point for the stitching
        t_p_zyx_residual_shifts = np.round((t_p_zyx_fourier_translations - t_zyx_global_shifts[:, None, :])).astype(np.int)

        #pcache these values so they don't need to be recomputed
        np.savez('{}{}_optimized_params'.format(param_cache_dir, output_basename + '_time_reg'),
                 **{'t_zyx_global_shifts': t_zyx_global_shifts, 't_p_zyx_residual_shifts': t_p_zyx_residual_shifts})

    ##### Stitching loop #####
    #default to 0 if not loading or computing
    p_zyx_stitch = np.zeros((metadata['num_positions'], 3), dtype=np.int)

    param_cache_name = output_basename
    saved_name = '{}{}_optimized_stitch_params.npz'.format(param_cache_dir, param_cache_name)
    if os.path.isfile(saved_name) and load_params:
        with np.load(saved_name) as loaded:
            if 'p_zyx_stitch' in loaded:
                print('Loaded params from: ' + saved_name)
                p_zyx_stitch = loaded['p_zyx_stitch']

    if stitch:
        print('Stitching timepoints')
        # for frame_index in range(min_tp, max_tp):

        stacks_nonempty_timestamps = [read_raw_data(magellan, metadata, time_index=time_index, reverse_rank_filter=reverse_rank_filter,
                       input_filter_sigma=input_filter_sigma) for time_index in range(max_tp - min_tp)]
        t_p_zyxc_stacks, t_p_nonempty_pixels, timestamps = list(zip(*stacks_nonempty_timestamps))

        if backgrounds is None:
            # get backgrounds from first time point
            backgrounds = estimate_background(t_p_zyxc_stacks[0], t_p_nonempty_pixels[0])

        total_shifts = -t_zyx_global_shifts[:, None,:] - t_p_zyx_residual_shifts
        #TODO: probably want to try flipping the sign of this and visualizing result
        t_p_z_positive_z_offset = total_shifts[:, :, 0] - np.min(total_shifts[:, :, 0])
        shifted_z_size = np.ptp(np.ravel(t_p_z_positive_z_offset)) + 1

        p_zyxc_preprocessed_stacks = {}
        for pos_index in t_p_zyxc_stacks[0].keys():
            time_series = []
            for time_index in range(len(t_p_zyxc_stacks)):
                p_zyxc_stacks = t_p_zyxc_stacks[time_index]
                #apply yx translation and get only the registration channels
                zyxc_registered_stack = np.stack([apply_intra_stack_registration(p_zyxc_stacks[pos_index][..., c],
                                t_p_yx_translations[time_index][pos_index] - t_p_zyx_residual_shifts[time_index][pos_index][1:],
                            background=np.mean(backgrounds[c]), mode='float') for c in range(p_zyxc_stacks[0].shape[3])], axis=3)

                registered_stack = np.zeros([shifted_z_size + zyxc_registered_stack.shape[0], zyxc_registered_stack.shape[1],
                                             zyxc_registered_stack.shape[2], zyxc_registered_stack.shape[3]])
                registered_stack[t_p_z_positive_z_offset[time_index, pos_index]:
                                 t_p_z_positive_z_offset[time_index, pos_index] +
                                                    zyxc_registered_stack.shape[0]] = zyxc_registered_stack
                time_series.append(registered_stack)

            zyxc_preprocessed_stack = np.mean(time_series, axis=0)
            #take min of autofluor channels, and use the full parts of others
            zyxc_preprocessed_stack = np.stack([np.min([zyxc_preprocessed_stack[..., c] for c in af_register_channels], axis=0)] + 
                                        [zyxc_preprocessed_stack[..., c] for c in other_register_channels], axis=3)

            p_zyxc_preprocessed_stacks[pos_index] = zyxc_preprocessed_stack



        if stitch_method == 'optimize':
            p_zyx_stitch = optimize_timepoint_stitching(p_zyxc_preprocessed_stacks,
                                              row_col_coords=metadata['row_col_coords'],
                                              overlap_shape=metadata['tile_overlaps'],
                                              p_yx_translations=None,
                                              p_zyx_intitial=None,
                                              inter_stack_channels=None,
                                              pixel_size_z=magellan.pixel_size_z_um,
                                              pixel_size_xy=magellan.pixel_size_xy_um,
                                              stitch_downsample_factor_xy=stitch_downsample_factor_xy,
                                              stitch_regularization_xy=0,
                                              stitch_regularization_z=0)


        # elif 'fourier' in stitch_method:
        #     if backgrounds is None:
        #         backgrounds = estimate_background(tp0_zyxc_stacks, nonempty_pixels)
        #     p_zyx_stitch_xys = compute_inter_stack_registrations(tp0_zyxc_stacks, nonempty_pixels, p_yx_translations,
        #                                                      tp0_p_zyx_residual_shifts, metadata,
        #                                                        channel_indices=xy_register_channels,
        #                                                        backgrounds=backgrounds, invert=(stitch_method[-1] == '1' or stitch_method[-1] == '3'))
        #     p_zyx_stitch_zs = compute_inter_stack_registrations(tp0_zyxc_stacks, nonempty_pixels, p_yx_translations,
        #                                                          tp0_p_zyx_residual_shifts, metadata,
        #                                                          channel_indices=z_register_channels,
        #                                                          backgrounds=backgrounds, invert=(stitch_method[-1] == '1' or stitch_method[-1] == '3'))
        #     p_zyx_stitch = np.concatenate([p_zyx_stitch_zs[:, 0][: ,None], p_zyx_stitch_xys[:, 1:]], axis=1)

        # cache these values so they don't need to be recomputed
        np.savez('{}{}_optimized_stitch_params'.format(param_cache_dir, output_basename), **{'p_zyx_stitch': p_zyx_stitch})


    #merge stitch and other one
    t_p_zyx_translations = np.round(t_p_zyx_residual_shifts + p_zyx_stitch).astype(np.int)
    t_p_zyx_translations[:, :, 0] -= np.min(t_p_zyx_translations[:, :, 0])

    if not export:
        return

    #make all global shifts positive

    t_zyx_global_shifts -= np.min(t_zyx_global_shifts, axis=0)
    t_zyx_global_shifts = np.round(t_zyx_global_shifts).astype(np.int)

    #compute the size of teh stiched image accounting for movements in z
    stitched_image_size = [
        np.ptp(np.reshape(t_p_zyx_translations, [-1, 3])[0]) + metadata['max_z_index'] - metadata['min_z_index'] + 1,
        (1 + np.ptp(metadata['row_col_coords'][:, 0], axis=0)) * (
                    metadata['tile_shape'][0] - metadata['tile_overlaps'][0]),
        (1 + np.ptp(metadata['row_col_coords'][:, 1], axis=0)) * (
                    metadata['tile_shape'][1] - metadata['tile_overlaps'][1])]
    #add in time point to timepoint registrations for the final imaris size
    imaris_size = np.array(stitched_image_size) + np.max(t_zyx_global_shifts, axis=0).astype(np.int) + \
                    np.max(t_p_zyx_translations[:, :, 0]).astype(np.int)


    output_basename = output_basename + suffix
    stitch_register_imaris_write(output_dir, output_basename, imaris_size, max_tp - min_tp, magellan, metadata, t_p_yx_translations,
                                    t_p_zyx_translations, t_zyx_global_shifts, input_filter_sigma=input_filter_sigma,
                                 reverse_rank_filter=reverse_rank_filter)
