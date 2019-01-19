from pygellan import MagellanDataset
from imariswriter import ImarisJavaWrapper
import numpy as np
from scipy.ndimage import filters
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi

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

def phase_correlate(src_image, target_image, use_unnormalized=True):
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
        cross_corr = np.fft.ifftn(normalized_cross_power_spectrum)
    maxima = np.array(np.unravel_index(np.argmax(np.abs(np.fft.fftshift(cross_corr))), cross_corr.shape))
    shifts = np.array(maxima, dtype=np.float)
    shifts -= np.array(cross_corr.shape) / 2
    return shifts

def smooth_and_register(stack, z_index, channel_index, sigma=6):
    """
    gaussian smooth, then compute pairwise registration
    :param stack:
    :param z_index:
    :param channel_index:
    :param sigma:
    :return:
    """
    current_img = stack[channel_index][z_index]
    prev_img = stack[channel_index][z_index - 1]
    filt1 = filters.gaussian_filter(current_img.astype(np.float), sigma)
    filt2 = filters.gaussian_filter(prev_img.astype(np.float), sigma)
    offset = phase_correlate(filt1, filt2)
    return offset

def compute_registrations(stack, nonempty):
    #tuple with shifts for optimal registartion
    registrations = []
    #booleans whether registrations for two slices was actuall computed form data
    valid_registrations = []
    #compute registrations for each valid set of consecutive slices
    for channel_index in range(len(list(stack.keys()))):
        registrations.append([])
        for z_index in range(len(stack[channel_index])):
            if channel_index == 0:
                valid_registrations.append(False)
            if z_index == 0:
                # take first one as origin
                registrations[channel_index].append((0, 0))
            elif (not nonempty[channel_index][z_index - 1]) or (not nonempty[channel_index][z_index]):
                #only compute registration if data was acquired at both
                registrations[channel_index].append((0, 0))
            else:
                registrations[channel_index].append(smooth_and_register(stack, z_index, channel_index))
                valid_registrations[z_index] = True
    #convert from relative registrations to slice above to absolute
    #exclude violet channel, taking median shift from all other channels
    reg_arr = np.array(registrations)[1:]
    consensus_reg = []
    for slice in range(reg_arr.shape[1]):
        consensus_reg.append(np.median(reg_arr[:, slice, :], axis=0))
    consensus_reg = np.array(consensus_reg)
    abs_reg = np.cumsum(consensus_reg, axis=0)
    #subtract median absolute registration so registrations are relative to center of most pixels
    abs_reg -= np.median(abs_reg[valid_registrations], axis=0)
    #use whole integer pixel shifts
    abs_reg = np.round(abs_reg).astype(np.int)
    return abs_reg

def register_z_stack(stack, registrations):
    """
    Apply the computed within z-stack registrations to all channels
    :param stack: dict with channel indices as keys and 3D numpy arrays specific to a single stack in a single channel
    as values
    :param registrations: 2D registration vectors corresponding to each slice
    :return: a list of all channels with a registered stack in each
    """
    registered_stacks = []
    for channel_index in stack.keys():
        one_channel_registered_stack = np.zeros(stack[0].shape)
        for slice in range(registrations.shape[0]):
            # need to negate to make it work right
            reg = -registrations[slice]
            reg_slice = one_channel_registered_stack[slice, ...]
            orig_slice = stack[channel_index][slice, ...]
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
        registered_stacks.append(one_channel_registered_stack)
    return registered_stacks

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

def compute_stack_registrations(stacks, channel_index, sigma=3):
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

def stitch_image(raw_stacks, params, metadata):
    """
    Stitch raw stacks into
    :param raw_data: dict with positions as keys containing list with 1 3d numpy array of pixels for each channel
    :param params:
    :return:
    """
    all_yx_translations = np.concatenate(list(zip(*params))[1], axis=0)
    all_z_tranlations = np.array(list(zip(*params))[0])

    # translate image so top left corner doesn't have excess space
    yx_global_translation_offset = -np.min(all_yx_translations, axis=0) + np.array(
                            [metadata['tile_height'] // 2, metadata['tile_width'] // 2])
    z_global_translation_offset = -np.min(all_z_tranlations)
    # Figure out size of stitched image
    # image size is range between biggest and smallest translation + 1/2 tile size on either side
    stitched_image_size = [np.ptp(all_z_tranlations) + metadata['max_z_index'] - metadata['min_z_index'] + 1,
                     np.ptp(all_yx_translations, axis=0)[0] + metadata['tile_height'],
                     np.ptp(all_yx_translations, axis=0)[1] + metadata['tile_width']]
    stitched_all_channels = []
    for channel_index in range(len(raw_stacks[0])):
        stitched = np.zeros(stitched_image_size, dtype=np.uint8 if metadata['byte_depth'] == 1 else np.uint16)
        # go through each tile and add into the appropriate place
        for position_index in range(metadata['num_positions']):
            stack = raw_stacks[position_index][channel_index]
            z_offset = params[position_index][0] + z_global_translation_offset
            # TODO: maybe be smarter about picking valid parts of images
            for z_index in range(stack.shape[0]):
                yx_offset = params[position_index][1][z_index] + yx_global_translation_offset
                stitched[z_offset + z_index,
                    yx_offset[0] - metadata['tile_height'] // 2:yx_offset[0] + metadata['tile_height'] // 2,
                    yx_offset[1] - metadata['tile_height'] // 2:yx_offset[1] + metadata['tile_height'] // 2] = stack[z_index]
        stitched_all_channels.append(stitched)
    return stitched_all_channels

def initialize_optimization_params():
    # create initial values for optimization
    params = []
    # yx params represent distance to center, z represents distance from top
    for position_index in range(metadata['num_positions']):
        z = 0
        row, col = magellan.row_col_tuples[position_index]
        # create default alignment based on no stitching
        center_y = metadata['tile_height'] // 2 + row * (metadata['tile_height'] - metadata['overlap_y'])
        center_x = metadata['tile_width'] // 2 + col * (metadata['tile_width'] - metadata['overlap_x'])
        # one yx vector per z slice, even ones with no data (will be ignroed later)
        yx = raw_stacks[position_index][0].shape[0] * [(center_y, center_x)]
        params.append((z, np.array(yx)))
    return params

def compute_alignment_cost(stacks, nonempty_pixels, params):

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

magellan_dir = '/Users/henrypinkard/Desktop/Lymphosight/2018-6-2 4 hours post LPS/subregion timelapse_1'
imaris_dir = os.sep.join(magellan_dir.split(os.sep)[:-1]) #parent directory of magellan
imaris_name = magellan_dir.split(os.sep)[-1] #same name as magellan acquisition
magellan, metadata = open_magellan(magellan_dir)

#TODO remove later
metadata['num_frames'] = 1

raw_data = read_raw_data(magellan, metadata, reverse_rank_filter=True)
# time_series = []
# for raw_stacks, nonempty_pixels, elapsed_time_ms in raw_data:
#     params = compute_stack_registrations(raw_stacks, channel_index=5)
#     params = initialize_optimization_params()
#     cost = compute_alignment_cost(raw_stacks, nonempty_pixels, params)
#
#     time_series.append((stitch_image(raw_stacks, params, metadata), elapsed_time_ms))




#toy problem to test stitching
translation_offset = [40, 0]
sigma = 1.5
background = 10

stacks = raw_data[0][0]
img1 = stacks[2][0][43]
img2 = stacks[4][0][43]
img1[img1 < background] = background
img2[img2 < background] = background
img1 = ndi.gaussian_filter(img1, sigma)
img2 = ndi.gaussian_filter(img2, sigma)


def stitch(translation_offset):
    yx_translation = np.array([[0, 0], [translation_offset[0], translation_offset[1] + metadata['tile_width'] - metadata['overlap_x']]])
    # translate image so top left corner doesn't have excess space
    yx_global_translation_offset = -np.min(yx_translation, axis=0) + np.array(
                          [metadata['tile_height'] // 2, metadata['tile_width'] // 2])
    # Figure out size of stitched image
    # image size is range between biggest and smallest translation + 1/2 tile size on either side
    stitched_image_size = [np.ptp(yx_translation, axis=0)[0] + metadata['tile_height'],
                   np.ptp(yx_translation, axis=0)[1] + metadata['tile_width']]
    stitched = np.ones(stitched_image_size, dtype=np.uint8 if metadata['byte_depth'] == 1 else np.uint16) *background
    # go through each tile and add into the appropriate place
    yx_offsets = yx_translation + yx_global_translation_offset
    stitched[yx_offsets[0, 0] - metadata['tile_height'] // 2:yx_offsets[0, 0] + metadata['tile_height'] // 2,
            yx_offsets[0, 1] - metadata['tile_height'] // 2:yx_offsets[0, 1] + metadata['tile_height'] // 2] = img1
    stitched[yx_offsets[1, 0] - metadata['tile_height'] // 2:yx_offsets[1, 0] + metadata['tile_height'] // 2,
            yx_offsets[1, 1] - metadata['tile_height'] // 2:yx_offsets[1, 1] + metadata['tile_height'] // 2] = img2
    return stitched

cost = []
for offset in range(0, 60, 5):
    translation_offset = [offset, 0]
    stitched = stitch(translation_offset)
    #for making a mask
    y, x = np.meshgrid(np.arange(stitched.shape[0]), (np.arange(stitched.shape[1])))
    distance = (x.T - stitched.shape[1] / 2)**2 / (stitched.shape[1] ** 2) + (
                    y.T - stitched.shape[0] / 2)**2 / (stitched.shape[0] ** 2)
    mask = np.ones(distance.shape)
    mask[distance < 0.0002] = 0
    # plt.imshow(mask)
    # plt.show()
    mask = mask == 1
    ft = np.fft.fftshift(np.fft.fftn(stitched))
    absft = np.abs(ft)
    logmagft = np.log(absft)
    normed = absft / np.linalg.norm(np.ravel(absft))

    cost.append(np.sum(normed[mask]))
    # plt.subplot(211)
    # plt.imshow(stitched)
    # plt.subplot(212)
    # plt.imshow(logmagft)
    # plt.show()
plt.plot(cost)
plt.show()




#TODO: write regularizer
#TODO: write optimization routine (could maybe write out intermediate stitched images as optimization proceeds

# write_imaris(imaris_dir, imaris_name + '_no_pos_reg_stitched_no_registration', time_series, metadata)


#Write out a single position to Imaris
# write_imaris(imaris_dir, imaris_name + '_single_pos_registered', [(registered_stacks[0], elapsed_time_ms)], metadata)
# write_imaris(imaris_dir, imaris_name + '_single_pos_unregistered', [(raw_stacks[0], elapsed_time_ms)], metadata)



#TODO: 3) register entire volume to itself from time point to time point


#TODO: stitching code, in case you need it later
# for position_index in range(num_positions):
#     # row, col = coords_by_pos_indices[position_index]
#     row, col = magellan.row_col_tuples[position_index]
#     if not magellan.has_image(channel_index=channel_index, pos_index=position_index,
#                               z_index=z_index, t_index=time_index):
#         continue
#     image, metadata = magellan.read_image(channel_index=channel_index, pos_index=position_index,
#                                           z_index=z_index, t_index=time_index, read_metadata=True)
#     central_square = image[overlap_y // 2:-overlap_y // 2, overlap_x // 2:-overlap_x // 2]
#     # add in central square to stitched image
#     stitched_image[row * (tile_height - overlap_y):(row + 1) * (tile_height - overlap_y), col *
#                             (tile_width - overlap_x):(col + 1) * (tile_width - overlap_x)] = central_square


# #compute dimesnions of stitched image
# imaris_size_x = metadata['num_cols'] * (metadata['tile_width'] - metadata['overlap_x'])
# imaris_size_y = metadata['num_rows'] * (metadata['tile_height'] - metadata['overlap_y'])
# imaris_size_z = metadata['max_z_index'] - metadata['min_z_index'] + 1

