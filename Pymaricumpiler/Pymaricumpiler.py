from pygellan import MagellanDataset
from imariswriter import ImarisJavaWrapper
import numpy as np
from scipy.ndimage import filters
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from scipy import signal
from itertools import combinations


def open_magellan(path):
    """
    open a magellan dataset on disk and read all appropriate metadata fields
    :param path: path to top level magellan folder
    :return:
    """
    magellan = MagellanDataset(path)
    metadata = {}
    #read metadata
    if magellan.summary_metadata['PixelType'] == 'GRAY8':
        metadata['byte_depth'] = 1
    else:
        metadata['byte_depth'] = 2
    metadata['num_positions'] = magellan.get_num_xy_positions()
    min_z_index, max_z_index = magellan.get_min_max_z_index()
    metadata['min_z_index'] = min_z_index
    metadata['max_z_index'] = max_z_index
    metadata['num_channels'] = len(magellan.summary_metadata['ChNames'])
    metadata['overlap_x'] = magellan.summary_metadata['GridPixelOverlapX']
    metadata['overlap_y'] = magellan.summary_metadata['GridPixelOverlapY']
    metadata['tile_width'] = magellan.summary_metadata['Width']
    metadata['tile_height'] = magellan.summary_metadata['Height']
    metadata['pixel_size_xy_um'] = magellan.summary_metadata['PixelSize_um']
    metadata['pixel_size_z_um'] = magellan.summary_metadata['z-step_um']
    metadata['num_frames'] = magellan.get_num_frames()
    num_rows, num_cols = magellan.get_num_rows_and_cols()
    metadata['num_rows'] = num_rows
    metadata['num_cols'] = num_cols
    return magellan, metadata

def read_raw_data(magellan, metadata, reverse_rank_filter=False):
    """
    read raw data, store in 3D arrays for each channel at each position
    :param magellan:
    :param metadata:
    :param reverse_rank_filter:
    :return:
    """
    time_series = []
    for time_index in range(metadata['num_frames']):
        elapsed_time_ms = ''
        raw_stacks = {}
        nonempty_pixels = {}
        for position_index in range(metadata['num_positions']):
            raw_stacks[position_index] = {}
            nonempty_pixels[position_index] = {}
            for channel_index in range(metadata['num_channels']):
                print('building channel {}, position {}'.format(channel_index, position_index))
                raw_stacks[position_index][channel_index] = np.zeros((metadata['max_z_index'] -
                        metadata['min_z_index'] + 1, metadata['tile_width'], metadata['tile_height']),
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
                    #TODO: make background pixel values?
        time_series.append((raw_stacks, nonempty_pixels, elapsed_time_ms))
    return time_series

def phase_correlate(src_image, target_image, use_unnormalized=True, max_shift=None):
    """
    Compute ND registration between two images
    :param src_image:
    :param target_image:
    :return:
    """
    src_ft = np.fft.fftn(src_image)
    target_ft = np.fft.fftn(target_image)
    if use_unnormalized == True:
        cross_corr = np.fft.ifftn((src_ft * target_ft.conj()))
    else:
        normalized_cross_power_spectrum = (src_ft * target_ft.conj()) / np.abs(src_ft * target_ft.conj())
        normalized_cross_corr = np.fft.ifftn(normalized_cross_power_spectrum)
        cross_corr = normalized_cross_corr
    cross_corr_mag = np.abs(np.fft.fftshift(cross_corr))
    if max_shift == None:
        np.min(np.array(cross_corr.shape)) // 2
    search_offset = (np.array(cross_corr.shape) // 2 - int(max_shift)).astype(np.int)
    shifts = np.array(np.unravel_index(np.argmax(
            cross_corr_mag[search_offset[0]:search_offset[0] + 2 * int(max_shift),
                            search_offset[1]:search_offset[1] + 2 * int(max_shift)]), (2 *int(max_shift), 2 *int(max_shift)) ))
    shifts += search_offset
    return shifts.astype(np.float) - np.array(cross_corr.shape) / 2

def register_z_stack(channel_stack, registrations, background=0):
    """
    Apply the computed within z-stack registrations to all channels
    :param stack: dict with channel indices as keys and 3D numpy arrays specific to a single stack in a single channel
    as values
    :param registrations: 2D registration vectors corresponding to each slice
    :return: a list of all channels with a registered stack in each
    """
    registered_stacks = []
    for channel_index in range(len(channel_stack)):
        one_channel_registered_stack = np.zeros(channel_stack[0].shape)
        for slice in range(registrations.shape[0]):
            one_channel_registered_stack[slice, ...] = ndi.shift(channel_stack[channel_index][slice],
                                                                 registrations[slice], cval=background)
            one_channel_registered_stack[one_channel_registered_stack < background] = background
        registered_stacks.append(one_channel_registered_stack)
    return registered_stacks

def intra_stack_registrations(channel_stack, nonempty_pixels, max_shift=20, background=10, use_channels=[1, 2, 3, 4, 5],
                              agreement_k=3, likelihood_agreement_threshold=7e-3, sigma_noise=2):
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

    def intra_stack_x_corr_regs(stack, nonempty, use_unnormalized=True, sigma=6, max_shift=None):
        """
        Smooth and cross correlate successive slices then take cumulative sum to figure out relative registrations for all
        channels within a stack
        """

        def smooth_and_register(stack, z_index, channel_index, sigma=6, use_unnormalized=True, max_shift=None):
            """
            gaussian smooth, then compute pairwise registration
            """
            current_img = stack[channel_index][z_index]
            prev_img = stack[channel_index][z_index - 1]
            filt1 = filters.gaussian_filter(current_img.astype(np.float), sigma)
            filt2 = filters.gaussian_filter(prev_img.astype(np.float), sigma)
            offset = phase_correlate(filt1, filt2, use_unnormalized=use_unnormalized, max_shift=max_shift)
            return offset

        # tuple with shifts for optimal registartion
        regs = []
        # compute registrations for each valid set of consecutive slices
        for channel_index in range(len(list(stack.keys()))):
            regs.append([])
            for z_index in range(len(stack[channel_index])):
                if z_index == 0:
                    # take first one as origin
                    regs[channel_index].append((0, 0))
                elif (not nonempty[z_index - 1]) or (not nonempty[z_index]):
                    # only compute registration if data was acquired at both
                    regs[channel_index].append((0, 0))
                else:
                    offset = smooth_and_register(stack, z_index, channel_index, use_unnormalized=use_unnormalized,
                                                 sigma=sigma, max_shift=max_shift)
                    regs[channel_index].append(offset)
        abs_regs = []
        for channel_reg in regs:
            abs_regs.append(np.cumsum(channel_reg, axis=0))
        return abs_regs

    # make a copy and set background
    channel_stack = {key: channel_stack[key].copy() for key in channel_stack.keys()}
    for channel in channel_stack.values():
        channel[channel < background] = background
    absolute_registrations = intra_stack_x_corr_regs(channel_stack, nonempty_pixels, sigma=2, max_shift=max_shift)
    zero_centered_regs = []

    registration_background_subtract_sigma = 2.5
    for channel_reg in absolute_registrations:
        background_shift = np.array([
                ndi.filters.gaussian_filter1d(channel_reg[:, 0], sigma=registration_background_subtract_sigma),
                ndi.filters.gaussian_filter1d(channel_reg[:, 1], sigma=registration_background_subtract_sigma)]).T
        zero_centered_regs.append(channel_reg - background_shift)
    # plt.figure(); plt.plot(np.array(absolute_registrations)[1:,:, 0], '.-')
    # plt.figure(); plt.plot(np.array(zero_centered_regs)[1:,:, 0].T); plt.ylim([-24, 24]); plt.legend([str(i) for i in range(5)])

    #ignore empty slices
    regs_to_use = np.array([zero_centered_regs[channel] for channel in use_channels])[..., nonempty_pixels, :]

    ####### compute per-chanel likelihoods #########
    x = np.linspace(-max_shift, max_shift, 500)
    #(channels) x (z slice) x (2 dims of registration) x (parameter space)
    likelihoods = np.zeros((regs_to_use.shape[:2]) + (2, x.size))
    for channel_index in range(regs_to_use.shape[0]):
        for z_index in range(regs_to_use.shape[1]):
            registration = regs_to_use[channel_index, z_index]
            likelihoods[channel_index, z_index, :, :] = (1 / (np.sqrt(2*np.pi) * sigma_noise) *
                np.exp(-((np.stack(2*[x]) - np.expand_dims(registration, axis=1)) ** 2) / (2 * sigma_noise ** 2)))

    ##### Look for different channels that have similar likelihoods ######
    #figure out if at least k channels agree on a registration
    channel_comb_indices = [np.array(comb) for comb in combinations(range(regs_to_use.shape[0]), agreement_k)]
    channel_comb_mls = np.zeros((len(channel_comb_indices), *likelihoods.shape[1:-1]))
    for cc_index, channel_comb in enumerate(channel_comb_indices):
        channel_comb_mls[cc_index, :, :] = np.max(np.prod(likelihoods[channel_comb, :, :, :], axis=0), axis=2)
    best_k_channel_likelihood = np.max(channel_comb_mls, axis=0)
    best_k_channel_index = np.argmax(channel_comb_mls, axis=0)
    #just use the best channel trio from one channel arbitrarily
    likelihood_agreement_channels = np.array(channel_comb_indices)[best_k_channel_index][:, 0, :]
    likelihood_agreement_valid_slices = ndi.gaussian_filter1d(np.mean(best_k_channel_likelihood, axis=1), 2) > likelihood_agreement_threshold
    # plt.figure(); plt.semilogy(ndi.gaussian_filter1d(np.mean(best_k_channel_likelihood, axis=1), 2), '.-')

    ##### Compute MLE over only the best slices
    mles = np.zeros(likelihoods.shape[1:3])
    for z_index in range(likelihoods.shape[1]):
        if not likelihood_agreement_valid_slices[z_index]:
            #skip over slices where not enough channels predict roughly the same thing
            continue #keep estimates at 0
        else:
            likelihood_prod_best_channels = np.prod(likelihoods[likelihood_agreement_channels[z_index], z_index, :, :], axis=0)
            # likelihood_prod_all_channels = np.prod(likelihoods[:, z_index, :, :], axis=0)
            #MLE
            # mles[z_index] = x[np.argmax(likelihood_prod_all_channels, axis=1)]
            mles[z_index] = x[np.argmax(likelihood_prod_best_channels, axis=1)]
    # plt.figure(); plt.plot(mles, '.-')

    registrations = np.zeros(absolute_registrations[0].shape)
    registrations[nonempty_pixels] = mles
    # registered_stacks = register_z_stack(channel_stack, -registrations)
    return registrations

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

def write_imaris(directory, name, time_series, metadata):
    timepoint0 = time_series[0][0]
    num_channels = len(timepoint0)
    t0c0 = timepoint0[0]
    imaris_size_x = t0c0.shape[2]
    imaris_size_y = t0c0.shape[1]
    imaris_size_z = t0c0.shape[0]
    num_frames = len(time_series)

    byte_depth = metadata['byte_depth']
    #extract other metadata
    pixel_size_xy_um = metadata['pixel_size_xy_um']
    pixel_size_z_um = metadata['pixel_size_z_um']

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

def stitch_image(raw_stacks, translations, registrations, metadata, row_col_coords, background=None):
    """
    Stitch raw stacks into
    :param raw_data: dict with positions as keys containing list with 1 3d numpy array of pixels for each channel
    :param params:
    :return:
    """
    tile_shape = np.array([metadata['tile_height'], metadata['tile_width']])
    tile_overlap = np.array([metadata['overlap_y'], metadata['overlap_x']])
    registrations = np.round(registrations).astype(np.int)
    # translate image so top left corner doesn't have excess space
    #TODO: make sure z coordinates are done correctly, and that minium z offset is 0 so extra padding doesn't get added
    z_global_translation_offset = -np.min(translations[:, 0])
    # Figure out size of stitched image
    # image size is range between biggest and smallest translation + 1/2 tile size on either side
    stitched_image_size = [np.ptp(translations[:, 0]) + metadata['max_z_index'] - metadata['min_z_index'] + 1,
                           (1 + np.ptp(row_col_coords[:, 0], axis=0)) * (tile_shape[0] - tile_overlap[0]),
                           (1 + np.ptp(row_col_coords[:, 1], axis=0)) * (tile_shape[1] - tile_overlap[1])]
    stitched = []
    for channel_index in range(len(raw_stacks[0])):
        if background is not None:
            stitched.append(background * np.ones(stitched_image_size,
                           dtype=np.uint8 if metadata['byte_depth'] == 1 else np.uint16))
        else:
            stitched.append(np.zeros(stitched_image_size, dtype=np.uint8 if metadata['byte_depth'] == 1 else np.uint16))

    for z in np.arange(stitched[0].shape[0]):
        print('stitching slice {}'.format(z))
        transformed_zs = z + translations[:, 0]
        slice_regs = registrations[range(translations.shape[0]), transformed_zs[range(translations.shape[0])], :]
        tile_centers = translations[:, 1:]

        #add in each tile to appropriate place in stitched image
        for p_index in range(len(tile_centers)):
            #compute destination coordinates, and coordinates in tile to extact
            #destination coordinates are fixed
            destination_corners = np.array([row_col_coords[p_index] * (tile_shape - tile_overlap),
                (row_col_coords[p_index] + 1) * (tile_shape - tile_overlap) ])
            destination_size = tile_shape - tile_overlap
            #TODO: add in global translation? and check that it doesnt exceed overlap over 2 with the slice registration
            border_size = tile_overlap // 2 + slice_regs[p_index]
            for channel_index in range(len(raw_stacks[0])):
                stitched[channel_index][z, destination_corners[0, 0]:destination_corners[1, 0],
                                           destination_corners[0, 1]:destination_corners[1, 1]] = \
                                            raw_stacks[p_index][channel_index][transformed_zs[p_index],
                                            border_size[0]:border_size[0] + destination_size[0],
                                            border_size[1]:border_size[1] + destination_size[1]]
    return stitched

def compute_inter_stack_registrations(stacks, channel_index, sigma=3):
    """
    Register stacks to one another using phase correlation and a least squares fit
    :param stacks:
    :param channel_index:
    :return:
    """
    #Calculate pairwise correspondences by phase correlation for all adjacent tiles
    two_tile_registrations = []
    for position_index1 in range(metadata['num_positions']):
        for position_index2 in range(position_index1):
            #check if the two tiles are adjacent, and if so calcualte their phase correlation
            row1, col1 = magellan.row_col_tuples[position_index1]
            row2, col2 = magellan.row_col_tuples[position_index2]
            stack1 = stacks[position_index1][channel_index]
            stack2 = stacks[position_index2][channel_index]
            for i, img in enumerate(stack1):
                stack1[i, ...] = ndi.gaussian_filter(img, sigma=sigma)
            for i, img in enumerate(stack2):
                stack2[i, ...] = ndi.gaussian_filter(img, sigma=sigma)
            # exporttiffstack(stack1, "stack1filt")
            # exporttiffstack(stack2, "stack2filt")
            #TODO: probably want to guassian smooth
            #TODO: exclude tiles that don't seem to register correctly, maybe by comparing average and max displacement
            if row1 == row2 + 1 and col1 == col2:
                # offset = phase_correlate(stack1, stack2)
                expected_offset = np.array([0, metadata['tile_height'] - metadata['overlap_y'], 0])
            elif row1 == row2 and col1 == col2 + 1:
                # offset = phase_correlate(raw_stacks[channel_index])
                expected_offset = np.array([0, 0, metadata['tile_width'] - metadata['overlap_x']])
            else:
                continue #nonadjacent tiles
            two_tile_registrations.append((expected_offset, position_index2, position_index1))
            # print('{},{}   to   {},{}:   {}'.format(row1, col1, row2, col2, expected_offset))

    def least_squares_traslations(two_tile_registrations, metadata):
        #Put into least squares matrix to solve for tile translations up to additive constant
        # set absolute translations for position 0 equal to zero to define absolut coordiante system
        A = np.zeros((3, 3 * metadata['num_positions']))
        A[0, 0] = 1
        A[1, 1] = 1
        A[2, 2] = 1
        b = [0, 0, 0]
        for two_tile_registration, pos2, pos1 in two_tile_registrations:
            b.extend(two_tile_registration)
            a = np.zeros((3, 3*metadata['num_positions']))
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
        #TODO is changing the sign needed when moving to phase correlation?
        global_translations = -np.reshape(np.round(x), ( -1, 3)).astype(np.int)
        #Use global translations to stitch together timepoint into full volume
        #gloabal_translations is in z, y,x format
        #make all translations positive
        global_translations -= np.min(global_translations, axis=0)
        return global_translations

    return least_squares_traslations(two_tile_registrations, metadata)

def inter_stack_correlation_cost(stacks, nonempty_pixels, params):

    def compute_cost(stack1, stack2, yx_offset, z_index):
        img1 = stack1[z_index]
        img2 = stack2[z_index]
        #smooth both
        smoothed1 = ndi.gaussian_filter(img1.astype(np.float), sigma=2)
        smoothed2 = ndi.gaussian_filter(img2.astype(np.float), sigma=2)
        #extract strips
        if yx_offset[0] > 0:
            strip1 = smoothed1[:-yx_offset[0], :]
            strip2 = smoothed2[yx_offset[0]:, :]
        elif yx_offset[0] < 0:
            strip1 = smoothed1[-yx_offset[0]:, :]
            strip2 = smoothed2[:yx_offset[0], :]
        else:
            strip1 = smoothed1
            strip2 = smoothed2
        if yx_offset[1] > 0:
            strip1 = strip1[:, :-yx_offset[1]]
            strip2 = strip2[:, yx_offset[1]:]
        elif yx_offset[1] < 0:
            strip1 = strip1[:, -yx_offset[1]:]
            strip2 = strip2[:, :yx_offset[1]]
        #compute cost over strip pixels
        cost = np.sum(strip1 * strip2) / np.sqrt(np.sum(strip1**2)*np.sum(strip2**2)) / strip1.size
        return cost

    cost = 0
    for position_index1 in range(metadata['num_positions']):
        for position_index2 in range(position_index1):
            # check if the two tiles are adjacent
            row1, col1 = magellan.row_col_tuples[position_index1]
            row2, col2 = magellan.row_col_tuples[position_index2]
            for channel_index in range(metadata['num_channels']):
                stack1 = stacks[position_index1][channel_index]
                stack2 = stacks[position_index2][channel_index]
                for z_index in range(stack1.shape[0]):
                    if (not nonempty_pixels[position_index1][z_index]) or (not nonempty_pixels[position_index2][z_index]):
                        continue #only caclculate from real data
                    if row1 == row2 + 1 and col1 == col2:
                        yx_offset = params[position_index1][1][z_index] - params[position_index2][1][z_index]
                        cost += compute_cost(stack1, stack2, yx_offset, z_index)
                    elif row1 == row2 and col1 == col2 + 1:
                        yx_offset = params[position_index1][1][z_index] - params[position_index2][1][z_index]
                        cost += compute_cost(stack1, stack2, yx_offset, z_index)
                    else:
                        continue  # nonadjacent tiles
    return cost

def compute_fourier_cost_xy(stitched, hi_freq_cutoff=0.0002):
    """
    Compute cost based on relative amount of high frequency information in 2D stitched image
    :param stitched:
    :return:
    """
    # for making a mask
    y, x = np.meshgrid(np.arange(stitched.shape[0]), (np.arange(stitched.shape[1])))
    distance = (x.T - stitched.shape[1] / 2) ** 2 / (stitched.shape[1] ** 2) + (
            y.T - stitched.shape[0] / 2) ** 2 / (stitched.shape[0] ** 2)
    mask = np.ones(distance.shape)
    mask[distance < hi_freq_cutoff] = 0
    # plt.imshow(mask)
    # plt.show()
    mask = mask == 1
    ft = np.fft.fftshift(np.fft.fftn(stitched))
    absft = np.abs(ft)
    normed = absft / np.linalg.norm(np.ravel(absft))
    return np.sum(normed[mask])

def optimize_z_translations(raw_stacks, params, metadata, channel=0, search_range=10):
    """
    Add positions in 1 at a time, computing best offset
    """
    for next_pos_index in range(1, len(raw_stacks)):
        z_offset = params[next_pos_index][0]
        cost = []
        z_offsets = np.arange(z_offset - search_range, z_offset + search_range)
        for new_z in z_offsets:
            print('z_offset {}'.format(new_z))
            # use only some positions
            stacks_to_use = {key: raw_stacks[key] for key in raw_stacks.keys() if key <= next_pos_index}
            params[next_pos_index][0] = new_z
            image = stitch_image(stacks_to_use, params, metadata, background=10)
            stack_cost = 0
            for slice in image[channel]:
                stack_cost += compute_fourier_cost_xy(slice)
            cost.append(stack_cost)
        params[next_pos_index][0] = z_offsets[np.argmin(np.array(cost))]


magellan_dir = '/Users/henrypinkard/Desktop/Lymphosight/2018-6-2 4 hours post LPS/subregion timelapse_1'
imaris_dir = os.sep.join(magellan_dir.split(os.sep)[:-1]) #parent directory of magellan
imaris_name = magellan_dir.split(os.sep)[-1] #same name as magellan acquisition
magellan, metadata = open_magellan(magellan_dir)

#TODO remove later
metadata['num_frames'] = 1

raw_data = read_raw_data(magellan, metadata, reverse_rank_filter=True)
time_series = []
for raw_stacks, nonempty_pixels, elapsed_time_ms in raw_data:
    translation_params = np.zeros((metadata['num_positions'], 3), dtype=np.int)
    registration_params = []
    #estimate z offsets for each channel
    # optimize_z_params(raw_stacks, params, metadata)
    registered_stacks = {}
    #TODO: improve stitching so valid part of each image gets taken
    for position_index in raw_stacks.keys():
        print('Registering stack slices for position {}'.format(position_index))
        registration_params.append(intra_stack_registrations(raw_stacks[position_index],
                                                             nonempty_pixels[position_index], max_shift=max(metadata['overlap_x'], metadata['overlap_y'])))
    stitched = stitch_image(raw_stacks, translation_params, registration_params, metadata, np.array(magellan.row_col_tuples))
    time_series.append((stitched, elapsed_time_ms))


write_imaris(imaris_dir, imaris_name + '_tiles_registered', time_series, metadata)

    # Write out a single position to Imaris
    # write_imaris(imaris_dir, imaris_name + '_single_pos_registered', [(registered_stacks[0], elapsed_time_ms)], metadata)
    # write_imaris(imaris_dir, imaris_name + '_single_pos_unregistered', [(raw_stacks[0], elapsed_time_ms)], metadata)


# def fourier_experiement():
#     #toy problem to test stitching
#     translation_offset = [0, 0]
#     sigma = 1.5
#     background = 10
#
#     stacks = raw_data[0][0]
#     slice = 43
#     img1 = stacks[2][0][slice]
#     img2 = stacks[4][0][slice]
#     img1[img1 < background] = background
#     img2[img2 < background] = background
#     img1 = ndi.gaussian_filter(img1, sigma)
#     img2 = ndi.gaussian_filter(img2, sigma)
#
#     def stitch(translation_offset, img1, img2):
#         yx_translation = np.array([[0, 0], [translation_offset[0], translation_offset[1] + metadata['tile_width'] - metadata['overlap_x']]])
#         # translate image so top left corner doesn't have excess space
#         yx_global_translation_offset = -np.min(yx_translation, axis=0) + np.array(
#                               [metadata['tile_height'] // 2, metadata['tile_width'] // 2])
#         # Figure out size of stitched image
#         # image size is range between biggest and smallest translation + 1/2 tile size on either side
#         stitched_image_size = [np.ptp(yx_translation, axis=0)[0] + metadata['tile_height'],
#                        np.ptp(yx_translation, axis=0)[1] + metadata['tile_width']]
#         stitched = np.ones(stitched_image_size, dtype=np.uint8 if metadata['byte_depth'] == 1 else np.uint16) *background
#         # go through each tile and add into the appropriate place
#         yx_offsets = yx_translation + yx_global_translation_offset
#         stitched[yx_offsets[0, 0] - metadata['tile_height'] // 2:yx_offsets[0, 0] + metadata['tile_height'] // 2,
#                 yx_offsets[0, 1] - metadata['tile_height'] // 2:yx_offsets[0, 1] + metadata['tile_height'] // 2] = img1
#         stitched[yx_offsets[1, 0] - metadata['tile_height'] // 2:yx_offsets[1, 0] + metadata['tile_height'] // 2,
#                 yx_offsets[1, 1] - metadata['tile_height'] // 2:yx_offsets[1, 1] + metadata['tile_height'] // 2] = img2
#         return stitched
#
#     #try z translating
#
#     cost = []
#     for offset in range(-10, 10):
#         stitched = stitch(translation_offset, stacks[2][0][slice], stacks[4][0][slice + offset])
#         # for making a mask
#         y, x = np.meshgrid(np.arange(stitched.shape[0]), (np.arange(stitched.shape[1])))
#         distance = (x.T - stitched.shape[1] / 2) ** 2 / (stitched.shape[1] ** 2) + (
#                 y.T - stitched.shape[0] / 2) ** 2 / (stitched.shape[0] ** 2)
#         mask = np.ones(distance.shape)
#         mask[distance < 0.0002] = 0
#         # plt.imshow(mask)
#         # plt.show()
#         mask = mask == 1
#         ft = np.fft.fftshift(np.fft.fftn(stitched))
#         absft = np.abs(ft)
#         logmagft = np.log(absft)
#         normed = absft / np.linalg.norm(np.ravel(absft))
#
#         cost.append(np.sum(normed[mask]))
#         plt.imshow(stitched)
#         plt.show()
#     plt.plot(cost)
#     plt.show()
#
#     #try xy translating
#     cost = []
#     for offset in range(0, 60, 5):
#         translation_offset = [offset, 0]
#         stitched = stitch(translation_offset)
#         #for making a mask
#         y, x = np.meshgrid(np.arange(stitched.shape[0]), (np.arange(stitched.shape[1])))
#         distance = (x.T - stitched.shape[1] / 2)**2 / (stitched.shape[1] ** 2) + (
#                         y.T - stitched.shape[0] / 2)**2 / (stitched.shape[0] ** 2)
#         mask = np.ones(distance.shape)
#         mask[distance < 0.0002] = 0
#         # plt.imshow(mask)
#         # plt.show()
#         mask = mask == 1
#         ft = np.fft.fftshift(np.fft.fftn(stitched))
#         absft = np.abs(ft)
#         logmagft = np.log(absft)
#         normed = absft / np.linalg.norm(np.ravel(absft))
#
#         cost.append(np.sum(normed[mask]))
#         # plt.subplot(211)
#         # plt.imshow(stitched)
#         # plt.subplot(212)
#         # plt.imshow(logmagft)
#         # plt.show()
#     plt.plot(cost)
#     plt.show()




#TODO: 3) register entire volume to itself from time point to time point
