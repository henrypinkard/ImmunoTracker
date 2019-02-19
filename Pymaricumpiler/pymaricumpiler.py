from pygellan import MagellanDataset
from imariswriter import ImarisJavaWrapper
import numpy as np
from scipy.ndimage import filters
import os
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from scipy import signal, optimize
from itertools import combinations
import warnings
from joblib import Parallel, delayed


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

def read_raw_data(magellan, metadata, time_index, reverse_rank_filter=False):
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
                #add in image
                raw_stacks[position_index][channel_index][z_index] = image
                nonempty_pixels[position_index][z_index] = True
                if elapsed_time_ms == '':
                    elapsed_time_ms = image_metadata['ElapsedTime-ms']
    return raw_stacks, nonempty_pixels, elapsed_time_ms

def apply_intra_stack_registration(single_channel_stack, registrations, background=0, mode='integer'):
    """
    Apply the computed within z-stack registrations to all channels
    :param stack: dict with channel indices as keys and 3D numpy arrays specific to a single stack in a single channel
    as values
    :param registrations: 2D registration vectors corresponding to each slice
    :return: a list of all channels with a registered stack in each
    """

    if mode == 'float':
        one_channel_registered_stack = np.zeros(single_channel_stack.shape)
        for slice in range(registrations.shape[0]):
            one_channel_registered_stack[slice, ...] = ndi.shift(single_channel_stack[slice],
                                                                 -registrations[slice], cval=background)
            one_channel_registered_stack[one_channel_registered_stack < background] = background
        return one_channel_registered_stack
    else:
        stack_copy = single_channel_stack.copy()
        for slice in range(registrations.shape[0]):
            # need to negate to make it work right
            reg = -np.round(registrations).astype(np.int)[slice]
            reg_slice = stack_copy[slice, ...]
            orig_slice = stack_copy[slice, ...]
            if reg[0] > 0:
                reg_slice = reg_slice[reg[0]:, :]
                orig_slice = orig_slice[:-reg[0], :]
            elif reg[0] < 0:
                reg_slice = reg_slice[:reg[0], :]
                orig_slice = orig_slice[-reg[0]:, :]
            if reg[1] > 0:
                reg_slice = reg_slice[:, reg[1]:]
                orig_slice = orig_slice[:, :-reg[1]]
            elif reg[1] < 0:
                reg_slice = reg_slice[:, :reg[1]]
                orig_slice = orig_slice[:, -reg[1]:]
            reg_slice[:] = orig_slice[:]
        return stack_copy

def compute_intra_stack_registrations(raw_stacks, nonempty_pixels, max_shift, backgrounds,
            use_channels=[1, 2, 3, 4, 5], sigma_noise=2, abs_reg_bkgd_subtract_sigma=3,
                                      likelihood_threshold_smooth_sigma=1, valid_likelihood_threshold=-18):
    """
    Compute registrations for each z slice within a stack using method based on cross correaltion followed by gaussian
    filtering and MLE fitting
    :param channel_stack:
    :param nonempty_pixels:
    :param max_shift:
    :param background:
    :param use_channels:
    :param agreement_k:
    :param likelihood_agreement_threshold:
    :param sigma_noise:
    :return:
    """

    def intra_stack_x_corr_regs(stack, nonempty, max_shift=None):
        """
        Smooth and cross correlate successive slices then take cumulative sum to figure out relative registrations for all
        channels within a stack
        """

        def cross_correlation(src_image, target_image, dont_whiten=True, max_shift=None):
            """
            Compute ND registration between two images
            :param src_image:
            :param target_image:
            :param dont_whiten if, false, normalize before inverse transform (i.e. phase correlation)
            :return:
            """
            src_ft = np.fft.fftn(src_image)
            target_ft = np.fft.fftn(target_image)
            if dont_whiten == True:
                cross_corr = np.fft.ifftn((src_ft * target_ft.conj()))
            else:
                normalized_cross_power_spectrum = (src_ft * target_ft.conj()) / np.abs(src_ft * target_ft.conj())
                normalized_cross_corr = np.fft.ifftn(normalized_cross_power_spectrum)
                cross_corr = normalized_cross_corr
            cross_corr_mag = np.abs(np.fft.fftshift(cross_corr))
            if max_shift == None:
                max_shift = np.min(np.array(cross_corr.shape)) // 2
            search_offset = (np.array(cross_corr.shape) // 2 - int(max_shift)).astype(np.int)
            shifts = np.array(np.unravel_index(np.argmax(
                cross_corr_mag[search_offset[0]:search_offset[0] + 2 * int(max_shift),
                search_offset[1]:search_offset[1] + 2 * int(max_shift)]), (2 * int(max_shift), 2 * int(max_shift))))
            shifts += search_offset
            return shifts.astype(np.float) - np.array(cross_corr.shape) / 2

        def register(current_img, prev_img, max_shift=None, mode='xcorr'):
            """
            gaussian smooth, then compute pairwise registration
            """

            img1 = current_img.astype(np.float)
            img2 = prev_img.astype(np.float)
            if mode == 'xcorr':
                offset = cross_correlation(img1, img2, dont_whiten=True, max_shift=max_shift)
            elif mode == 'phase':
                offset = cross_correlation(img1, img2, dont_whiten=False, max_shift=max_shift)
            return offset

        # compute registrations for each valid set of consecutive slices
        def register_consecutives_slices(z_index, stack, nonempty, channel_index):
            if z_index == 0 or ((not nonempty[z_index - 1]) or (not nonempty[z_index])):
                # take first one as origin and only compute registration if data was acquired at both
                return (0, 0)
            else:
                current_img = stack[channel_index][z_index]
                prev_img = stack[channel_index][z_index - 1]
                return register(current_img, prev_img, max_shift=max_shift, mode='xcorr')

        regs = [[register_consecutives_slices(z_index, stack, nonempty, channel_index) for z_index in
                  range(len(stack[channel_index]))] for  channel_index in range(len(list(stack.keys())))]
        abs_regs = []
        for channel_reg in regs:
            abs_regs.append(np.cumsum(channel_reg, axis=0))
        return abs_regs

    registration_params = []
    for position_index in raw_stacks.keys():
        print('Registering stack slices for position {}'.format(position_index))
        channel_stack = raw_stacks[position_index]

        # make a copy and set background
        channel_stack = {key: channel_stack[key].copy() for key in channel_stack.keys()}
        for channel_index in range(len(channel_stack)):
            channel_stack[channel_index][channel_stack[channel_index] < backgrounds[channel_index]] = backgrounds[channel_index]
        absolute_registrations = intra_stack_x_corr_regs(
                                        channel_stack, nonempty_pixels[position_index], max_shift=max_shift)
        zero_centered_regs = []
        for channel_reg in absolute_registrations:
            background_shift = np.array([
                    ndi.filters.gaussian_filter1d(channel_reg[:, 0], sigma=abs_reg_bkgd_subtract_sigma),
                    ndi.filters.gaussian_filter1d(channel_reg[:, 1], sigma=abs_reg_bkgd_subtract_sigma)]).T
            zero_centered_regs.append(channel_reg - background_shift)
        # plt.figure(); plt.plot(np.array(absolute_registrations)[1:,:, 0].T, '.-')
        # plt.figure(); plt.plot(background_shift[:,0]); plt.show()
        # plt.figure(); plt.plot(np.array(zero_centered_regs)[1:,:, 0].T,'.-'); plt.ylim([-24, 24]); plt.legend([str(i) for i in range(5)])

        #ignore empty slices
        regs_to_use = np.array([zero_centered_regs[channel]
                                for channel in use_channels])[..., nonempty_pixels[position_index], :]
        # plt.figure(); plt.plot(regs_to_use[:,:, 0].T); plt.ylim([-24, 24]); plt.legend([str(i) for i in range(5)])

        ####### compute per-chanel likelihoods #########
        x = np.linspace(-max_shift, max_shift, 500)
        #(channels) x (z slice) x (2 dims of registration) x (parameter space)
        likelihoods = np.zeros((regs_to_use.shape[:2]) + (2, x.size))
        for channel_index in range(regs_to_use.shape[0]):
            for z_index in range(regs_to_use.shape[1]):
                registration = regs_to_use[channel_index, z_index]
                likelihoods[channel_index, z_index, :, :] = (1 / (np.sqrt(2*np.pi) * sigma_noise) *
                    np.exp(-((np.stack(2*[x]) - np.expand_dims(registration, axis=1)) ** 2) / (2 * sigma_noise ** 2)))

        ##### Compute MLE over only the best slices
        mles_all_channels = np.zeros(likelihoods.shape[1:3])
        mls_all_channels = np.zeros(likelihoods.shape[1:3])
        for z_index in range(likelihoods.shape[1]):
            likelihood_prod_all_channels = np.prod(likelihoods[:, z_index, :, :], axis=0)
            # plt.figure();  plt.semilogy(likelihood_prod_all_channels[0], '.-')
            #MLE
            mles_all_channels[z_index] = x[np.argmax(likelihood_prod_all_channels, axis=1)]
            mls_all_channels[z_index] = np.max(likelihood_prod_all_channels, axis=1)
        composite_log_likelihood = np.log(np.prod(mls_all_channels, axis=1))
        smoothed_log_likeliood = ndi.gaussian_filter1d(composite_log_likelihood, likelihood_threshold_smooth_sigma)
        # plt.figure(); plt.plot(smoothed_log_likeliood, '.-')
        valid_mles = smoothed_log_likeliood > valid_likelihood_threshold

        # mles_all_channels[np.logical_not(valid_mles)] = sin_pred_movement[np.logical_not(valid_mles)]
        mles_all_channels[np.logical_not(valid_mles)] = 0
        # plt.figure(); plt.plot(mles_all_channels)

        #apply the computed registrations to slices that have pixel data
        registrations = np.zeros(absolute_registrations[0].shape)
        registrations[nonempty_pixels[position_index]] = mles_all_channels

        # channel = 3
        # registered_stack = apply_intra_stack_registration(channel_stack[channel], registrations)
        # exporttiffstack(registered_stack, 'registered with sin channel {}'.format(channel))
        # exporttiffstack(channel_stack[channel], 'unregistered channel {}'.format(channel))
        registration_params.append(registrations)
    return registration_params

def exporttiffstack(datacube, name='export'):
    '''
    Save 3D numpy array as a TIFF stack
    :param datacube:
    '''
    if len(datacube.shape) == 2:
        imlist = [Image.fromarray(datacube)]
    else:
        imlist = []
        for i in range(datacube.shape[0]):
            imlist.append(Image.fromarray(datacube[i,...]))
    path = "/Users/henrypinkard/Desktop/{}.tif".format(name)
    imlist[0].save(path, compression="tiff_deflate", save_all=True, append_images=imlist[1:])

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

def stitch_single_channel(stacks, translations, registrations, tile_overlap, row_col_coords, channel_index,
                          backgrounds=None):
    """
    Stitch raw stacks into single volume
    :param raw_data: dict with positions as keys containing list with 1 3d numpy array of pixels for each channel
    :param params:
    :return:
    """
    #TODO: check row_col_coords
    stack_shape = stacks[0][0].shape
    byte_depth = 1 if stacks[0][0].dtype == np.uint8 else 2
    registrations = np.round(registrations).astype(np.int)
    # make z coordinate 0-based
    translations[:, 0] -= np.min(translations[:, 0])
    # Figure out size of stitched image
    # image size is range between biggest and smallest translation + 1/2 tile size on either side
    stitched_image_size = [np.ptp(translations[:, 0]) + stack_shape[0],
                   (1 + np.ptp(row_col_coords[:, 0], axis=0)) * (stack_shape[1] - tile_overlap[0]),
                   (1 + np.ptp(row_col_coords[:, 1], axis=0)) * (stack_shape[2] - tile_overlap[1])]
    if backgrounds is not None:
        stitched = backgrounds[channel_index] * np.ones(stitched_image_size, dtype=np.uint8 if byte_depth == 1 else np.uint16)
    else:
        stitched = np.zeros(stitched_image_size, dtype=np.uint8 if byte_depth == 1 else np.uint16)

    def get_stitch_coords(stitched_z, p_index):
        stack_z = stitched_z + translations[p_index, 0]
        if stack_z >= stacks[p_index][0].shape[0]:
            return None, None, None, None  # the z registration puts things out of bounds
        intra_stack_reg = registrations[p_index, stack_z, :]
        # compute destination coordinates, and coordinates in tile to extact
        # destination coordinates are fixed
        destination_corners = np.array([row_col_coords[p_index] * (stack_shape[1:] - tile_overlap),
                                        (row_col_coords[p_index] + 1) * (stack_shape[1:] - tile_overlap)])
        destination_size = stack_shape[1:] - tile_overlap
        border_size = tile_overlap // 2 + intra_stack_reg - tile_center_translations[p_index]
        return stack_z, destination_corners, destination_size, border_size

    def pad_from_neighbor_tile(p_index, axis, border_size, inserted_tile):
        # take from row above or row below
        row_col = row_col_coords[p_index].copy()
        if border_size[axis] < 0:
            row_col[axis] -= 1
            strip_width_or_height = np.abs(border_size[axis])
        else:
            row_col[axis] += 1
            strip_width_or_height = border_size[axis] - tile_overlap[axis]
        #this one is the correct size to be stacked with the original tile
        extra_strip = np.zeros((strip_width_or_height, inserted_tile.shape[1]) if
                               axis == 0 else (inserted_tile.shape[0], strip_width_or_height))
        # check if bordering tile exists
        if np.any(np.logical_and(row_col_coords[:, 0] == row_col[0], row_col_coords[:, 1] == row_col[0])):
            neighbor_p_index = int(np.logical_and(row_col_coords[:, 0] == row_col[0],
                                                  row_col_coords[:, 1] == row_col[1]).nonzero()[0])
            neighbor_stack_z, neighbor_dest_corners, neighbor_dest_size, neighbor_border_size = get_stitch_coords(
                stitched_z, neighbor_p_index)
            if neighbor_stack_z is not None:
                # fill in pixels from other tile into extra strip
                strip_destination = extra_strip
                # get coordinates within the neighbor tile based on its relative position
                if axis == 0:
                    if border_size[0] < 0:
                        #add pixels to the bottom of this tile from the top of the other one

                        axis0_neighbor_tile_coords = np.array([neighbor_border_size[0] + neighbor_dest_size[0],
                                          neighbor_border_size[0] + neighbor_dest_size[0] + strip_width_or_height])
                    else:
                        axis0_neighbor_tile_coords = np.array([neighbor_border_size[0] - strip_width_or_height,
                                                                    neighbor_border_size[0]])
                    axis1_neighbor_tile_coords = np.array([neighbor_border_size[1],
                                                       neighbor_border_size[1] + neighbor_dest_size[1]])
                else:
                    if border_size[1] < 0:
                        #add pixels to the top of this tile from the bottom of the neighboring one

                        axis1_neighbor_tile_coords = np.array([neighbor_border_size[1] + neighbor_dest_size[1],
                                          neighbor_border_size[1] + neighbor_dest_size[1] + strip_width_or_height])
                    else:
                        axis1_neighbor_tile_coords = np.array([neighbor_border_size[1] - strip_width_or_height,
                                                                    neighbor_border_size[1]])
                    axis0_neighbor_tile_coords = np.array([neighbor_border_size[0],
                                                       neighbor_border_size[0] + neighbor_dest_size[0]])

                #check if neighboring strip is out of bounds and adjust if so
                if axis0_neighbor_tile_coords[1] > stack_shape[1]:
                    axis0_neighbor_tile_coords[axis0_neighbor_tile_coords[1] > stack_shape[1]] = stack_shape[1]
                    strip_destination = strip_destination[-(axis0_neighbor_tile_coords[1] - axis0_neighbor_tile_coords[0]):, :]
                if axis0_neighbor_tile_coords[0] < 0:
                    axis0_neighbor_tile_coords[axis0_neighbor_tile_coords[0] < 0] = 0
                    strip_destination = strip_destination[:(axis0_neighbor_tile_coords[1] - axis0_neighbor_tile_coords[0]), :]
                if axis1_neighbor_tile_coords[1] > stack_shape[2]:
                    axis1_neighbor_tile_coords[axis1_neighbor_tile_coords[1] > stack_shape[2]] = stack_shape[2]
                    strip_destination = strip_destination[:, -(axis1_neighbor_tile_coords[1] - axis1_neighbor_tile_coords[0]):]
                if axis1_neighbor_tile_coords[0] < 0:
                    axis1_neighbor_tile_coords[axis1_neighbor_tile_coords[0] < 0] = 0
                    strip_destination = strip_destination[:, (axis1_neighbor_tile_coords[1] - axis1_neighbor_tile_coords[0])]

                strip_destination[:, :] = stacks[neighbor_p_index][channel_index][neighbor_stack_z,
                                        axis0_neighbor_tile_coords[0]:axis0_neighbor_tile_coords[1],
                                        axis1_neighbor_tile_coords[0]:axis1_neighbor_tile_coords[1]]

        #add stuff from the other tile, or 0s if it didnt overlap, then recrop to correct shape
        original_size = inserted_tile.shape
        if border_size[axis] < 0:
            inserted_tile = np.concatenate((extra_strip, inserted_tile), axis=axis)[:original_size[0], :original_size[1]]
        else:
            inserted_tile = np.concatenate((inserted_tile, extra_strip), axis=axis)[-original_size[0]:, -original_size[1]:]

        return inserted_tile

    print('stitching channel {}'.format(channel_index))
    for stitched_z in np.arange(stitched.shape[0]):
        # print('stitching slice {}'.format(stitched_z))
        tile_center_translations = translations[:, 1:]
        #add in each tile to appropriate place in stitched image
        for p_index in range(len(stacks)):
            stack_z, destination_corners, destination_size, border_size = get_stitch_coords(stitched_z, p_index)
            if stack_z is None:
                continue #Z is out of bounds of the stack

            #take the valid part of the tile
            cropped_border_size = border_size.copy()
            cropped_border_size[cropped_border_size < 0] = 0
            cropped_border_size[cropped_border_size > tile_overlap] = tile_overlap[cropped_border_size > tile_overlap]

            tile_to_add = stacks[p_index][channel_index][stack_z,
                          cropped_border_size[0]:cropped_border_size[0] + destination_size[0],
                          cropped_border_size[1]:cropped_border_size[1] + destination_size[1]]

            #add in overlapping parts from other tiles if this tiel didn't fill the frame properly
            if border_size[0] < 0 or border_size[0] > tile_overlap[0]:
                tile_to_add = pad_from_neighbor_tile(p_index, axis=0, border_size=border_size, inserted_tile=tile_to_add)
            if border_size[1] < 0 or border_size[1] > tile_overlap[1]:
                tile_to_add = pad_from_neighbor_tile(p_index, axis=1, border_size=border_size, inserted_tile=tile_to_add)

            stitched[stitched_z, destination_corners[0, 0]:destination_corners[1, 0],
                destination_corners[0, 1]:destination_corners[1, 1]] = tile_to_add
    return stitched

def stitch_all_channels(stacks, translations, registrations, tile_overlap, row_col_coords, background=None):
    stitched = []
    for channel_index in range(len(stacks[0])):
        stitched.append(stitch_single_channel(stacks, translations, registrations, tile_overlap,
                                              row_col_coords, channel_index=channel_index, background=None))
    return stitched

def x_corr_register_3D(volume1, volume2, max_shift):
    src_ft = np.fft.fftn(volume1)
    target_ft = np.fft.fftn(volume2)
    cross_corr = np.fft.ifftn((src_ft * target_ft.conj()))
    cross_corr_mag = np.abs(np.fft.fftshift(cross_corr))
    search_offset = (np.array(cross_corr.shape) // 2 - max_shift).astype(np.int)
    search_volume = cross_corr_mag[search_offset[0]:search_offset[0] + 2 * max_shift[0],
                    search_offset[1]:search_offset[1] + 2 * max_shift[1],
                    search_offset[2]:search_offset[2] + 2 * max_shift[2]]
    shifts = np.array(np.unravel_index(np.argmax(search_volume), search_volume.shape)).astype(np.int)
    shifts -= np.array(search_volume.shape) // 2
    return shifts

def normalized_x_corr_register_3D(volume1, volume2, max_shift):
    score = np.zeros((2*max_shift[0]+1, 2*max_shift[1]+1, 2*max_shift[2]+1))
    t = np.arange(-max_shift[0], max_shift[0] + 1)
    u = np.arange(-max_shift[1], max_shift[1] + 1)
    v = np.arange(-max_shift[2], max_shift[2] + 1)
    t_indices1 = np.stack((np.max(np.stack((t, np.zeros(t.shape))), axis=0),
               np.min(np.stack((volume1.shape[0] + t, np.ones(t.shape)*volume1.shape[0])), axis=0))).astype(np.int)
    t_indices2 = np.stack((np.max(np.stack((-t, np.zeros(t.shape))), axis=0),
               np.min(np.stack((volume1.shape[0] - t, np.ones(t.shape)*volume1.shape[0])), axis=0))).astype(np.int)
    u_indices1 = np.stack((np.max(np.stack((u, np.zeros(u.shape))), axis=0),
               np.min(np.stack((volume1.shape[1] + u, np.ones(u.shape) * volume1.shape[1])), axis=0))).astype(np.int)
    u_indices2 = np.stack((np.max(np.stack((-u, np.zeros(u.shape))), axis=0),
               np.min(np.stack((volume1.shape[1] - u, np.ones(u.shape) * volume1.shape[1])), axis=0))).astype(np.int)
    v_indices1 = np.stack((np.max(np.stack((v, np.zeros(v.shape))), axis=0),
               np.min(np.stack((volume1.shape[2] + v, np.ones(v.shape) * volume1.shape[2])), axis=0))).astype(np.int)
    v_indices2 = np.stack((np.max(np.stack((-v, np.zeros(v.shape))), axis=0),
               np.min(np.stack((volume1.shape[2] - v, np.ones(v.shape) * volume1.shape[2])), axis=0))).astype(np.int)


    for t_i, ((t11, t12), (t21, t22)) in enumerate(zip(t_indices1.T, t_indices2.T)):
        for u_i, ((u11, u12), (u21, u22)) in enumerate(zip(u_indices1.T, u_indices2.T)):
            for v_i, ((v11, v12), (v21, v22)) in enumerate(zip(v_indices1.T, v_indices2.T)):
                vol1_use = volume1[t11:t12, u11:u12, v11:v12]
                vol2_use = volume2[t21:t22, u21:u22, v21:v22]
                v1_mean_sub = vol1_use - np.mean(vol1_use.astype(np.float))
                v2_mean_sub = vol2_use - np.mean(vol2_use.astype(np.float))
                x_corr = np.sum(v1_mean_sub * v2_mean_sub)
                normalization = np.sqrt(np.sum(v1_mean_sub ** 2) * np.sum(v2_mean_sub ** 2))
                score[t_i, u_i, v_i] = x_corr / normalization

    shifts = np.array(np.unravel_index(np.argmax(score), score.shape)).astype(np.int)
    shifts -= np.array(score.shape) // 2
    return shifts

def compute_inter_stack_registrations(stacks, nonempty_pixels, registrations, metadata,
                                      max_shift_z, channel_index, backgrounds, n_cores=8):
    """
    Register stacks to one another using phase correlation and a least squares fit
    :param stacks:
    :param channel_index:
    :return:
    """
    row_col_coords = metadata['row_col_coords']
    tile_overlaps = metadata['tile_overlaps']
    max_shift = np.array([max_shift_z, int(0.8 * tile_overlaps[0]), int(0.8 * tile_overlaps[1])]).astype(np.int)

    #Calculate pairwise correspondences by phase correlation for all adjacent tiles
    volumes_to_register = []
    registration_position_indices = []
    for position_index1 in range(len(stacks)):
        row1, col1 = row_col_coords[position_index1]
        stack1_reg_channel = apply_intra_stack_registration(stacks[position_index1][channel_index],
                                                registrations[position_index1], background=backgrounds[channel_index])
        for position_index2 in range(position_index1):
            row2, col2 = row_col_coords[position_index2]
            if not ((row1 == row2 + 1 and col1 == col2) or (row1 == row2 and col1 == col2 + 1)):
                continue #non adjacent tiles
            stack2_reg_channel = apply_intra_stack_registration(stacks[position_index2][channel_index],
                                                registrations[position_index2], background=backgrounds[channel_index])

            #use only areas that are valid for both
            both_nonempty = np.logical_and(nonempty_pixels[position_index1], nonempty_pixels[position_index2])
            stack1_valid = stack1_reg_channel[both_nonempty, :, :]
            stack2_valid = stack2_reg_channel[both_nonempty, :, :]

            if row1 == row2 + 1 and col1 == col2:
                overlap1 = stack1_valid[:, :tile_overlaps[0], :]
                overlap2 = stack2_valid[:, -tile_overlaps[0]:, :]
            elif row1 == row2 and col1 == col2 + 1:
                overlap1 = stack1_valid[:, :, :tile_overlaps[1]]
                overlap2 = stack2_valid[:, :, -tile_overlaps[1]:]
            volumes_to_register.append((overlap1, overlap2))
            registration_position_indices.append((position_index1, position_index2))

    with Parallel(n_jobs=n_cores) as parallel:
        pairwise_registrations = parallel(delayed(normalized_x_corr_register_3D)(overlaps[0], overlaps[1], max_shift) for
                                                  overlaps in volumes_to_register)
    # two_tile_registrations = [(normalized_x_corr_register_3D(overlap1, overlap2, max_shift), position_index1,
    #                 position_index2) for position_index1, position_index2, overlap1, overlap2 in volumes_to_register]

    def least_squares_traslations(two_tile_registrations, registration_position_indices):
        #Put into least squares matrix to solve for tile translations up to additive constant
        # set absolute translations for position 0 equal to zero to define absolut coordiante system
        A = np.zeros((3, 3 * len(stacks)))
        A[0, 0] = 1
        A[1, 1] = 1
        A[2, 2] = 1
        b = [0, 0, 0]
        for i in range(len(two_tile_registrations)):
            two_tile_registration = two_tile_registrations[i]
            pos1, pos2 = registration_position_indices[i]
            b.extend(two_tile_registration)
            a = np.zeros((3, 3*len(stacks)))
            a[0, pos2 * 3] = 1
            a[0, pos1 * 3] = -1
            a[1, pos2 * 3 + 1] = 1
            a[1, pos1 * 3 + 1] = -1
            a[2, pos2 * 3 + 2] = 1
            a[2, pos1 * 3 + 2] = -1
            A = np.concatenate((A, a), 0)
        b = np.array(b)
        #solve least squares problem
        x = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b)
        #make global translations indexed by position index
        global_translations = -np.reshape(np.round(x), ( -1, 3)).astype(np.int)
        #Use global translations to stitch together timepoint into full volume
        #gloabal_translations is in z, y,x format
        #make all z translations positive
        global_translations -= np.min(global_translations, axis=0)
        return global_translations

    ls_traslations = least_squares_traslations(pairwise_registrations, registration_position_indices)
    print(ls_traslations)
    # zero center translation params, since offset is arbitrary
    ls_traslations[:, 1:] -= np.round((np.max(ls_traslations[:, 1:], axis=0) + np.min(ls_traslations[:, 1:], axis=0)) / 2).astype(np.int)
    #invert xy translations so they work correctly
    ls_traslations[:, 1:] *= -1
    return ls_traslations

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
    return np.median(np.stack(all_pix.values()), axis=1)


def ram_efficient_stitch_register_imaris_write(directory, name, imaris_size, magellan, metadata,
                    registration_series, translation_series, abs_timepoint_registrations):
    num_channels = metadata['num_channels']
    num_frames = metadata['num_frames']
    byte_depth = metadata['byte_depth']
    print('Imaris file: {}'.format(name))
    print('Imaris directory {}'.format(directory))
    with ImarisJavaWrapper(directory, name, (int(imaris_size[2]), int(imaris_size[1]), int(imaris_size[0])), byte_depth,
                num_channels, num_frames, metadata['pixel_size_xy_um'], float(metadata['pixel_size_z_um'])) as writer:
        for time_index in range(num_frames):
            print('Frame {}'.format(time_index))
            raw_stacks, nonempty_pixels, timestamp = read_raw_data(
                magellan, metadata, time_index=time_index, reverse_rank_filter=True)
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

def convert(magellan_dir, do_intra_stack=True, do_inter_stack=True, do_timepoints=True,
            output_dir=None, output_basename=None, intra_stack_registration_channels=[1, 2, 3, 4, 5],
            intra_stack_noise_model_sigma=2, intra_stack_zero_center_sigma=3,
            intra_stack_likelihood_threshold_smooth=1.0, intra_stack_likelihood_threshold=-18,
            inter_stack_registration_channel=0, inter_stack_max_z=7, timepoint_registration_channel=0, n_cores=8):
    """

    :param magellan_dir: directory of magellan data to be converted
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
            raw_stacks, nonempty_pixels, timestamp = read_raw_data(
                                magellan, metadata, time_index=frame_index, reverse_rank_filter=True)
            if backgrounds is None:
                #get backgrounds from first time point
                backgrounds = estimate_background(raw_stacks, nonempty_pixels)

        # Intravatal breathing artifact correcttions within stack
        if do_intra_stack:
            registration_params = compute_intra_stack_registrations(raw_stacks, nonempty_pixels,
               np.max(metadata['tile_overlaps']), backgrounds=backgrounds, use_channels=intra_stack_registration_channels,
                 sigma_noise=intra_stack_noise_model_sigma, abs_reg_bkgd_subtract_sigma=intra_stack_zero_center_sigma,
                                        likelihood_threshold_smooth_sigma=intra_stack_likelihood_threshold_smooth,
                                                        valid_likelihood_threshold=intra_stack_likelihood_threshold)
        else:
            registration_params = [np.zeros((metadata['max_z_index'] - metadata['min_z_index'] + 1, 2))
                                    for position_index in range(metadata['num_positions'])]
        # XYZ stack misalignments
        if do_inter_stack:
            translation_params = compute_inter_stack_registrations(raw_stacks, nonempty_pixels, registration_params,
                            metadata, max_shift_z=inter_stack_max_z, channel_index=inter_stack_registration_channel,
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
                                               magellan, metadata, registration_series, translation_series,
                                               abs_timepoint_registrations)


# magellan_dir = '/Users/henrypinkard/Desktop/Lymphosight/2018-6-2 4 hours post LPS/subregion timelapse_1'
#
# convert(magellan_dir, do_intra_stack=True, do_inter_stack=True,
#         inter_stack_registration_channel=0, timepoint_registration_channel=0, n_cores=8)