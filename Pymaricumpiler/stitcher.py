import numpy as np
from intravital_stack import  apply_intra_stack_registration
from joblib import Parallel, delayed
from utility import x_corr_register_3D, normalized_x_corr_register_3D
import tensorflow as tf

def compute_inter_stack_registrations(stacks, nonempty_pixels, registrations, metadata,
            max_shift_z, channel_indices, backgrounds, n_cores=8, max_shift_percentage=0.9, normalized_x_corr=True):
    """
    Register stacks to one another using phase correlation and a least squares fit
    :param stacks:
    :param channel_indices:
    :return:
    """
    row_col_coords = metadata['row_col_coords']
    tile_overlaps = metadata['tile_overlaps']
    max_shift = np.array([max_shift_z, int(max_shift_percentage * tile_overlaps[0]),
            int(max_shift_percentage * tile_overlaps[1])]).astype(np.int)

    #Calculate pairwise correspondences by phase correlation for all adjacent tiles
    volumes_to_register = []
    registration_position_channel_indices = []
    for channel_index in channel_indices:
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
                registration_position_channel_indices.append((position_index1, position_index2, channel_index))

    if normalized_x_corr:
        shift_and_weight = lambda v1, v2, max_shift: (normalized_x_corr_register_3D(v1,v2,max_shift),
                                                min(np.mean(np.ravel(v1)), np.mean(np.ravel(v2))))
        with Parallel(n_jobs=n_cores) as parallel:
            pairwise_registrations_and_weights = parallel(delayed(shift_and_weight)(overlaps[0], overlaps[1], max_shift)
                                                    for overlaps in volumes_to_register)

        # pairwise_registrations_and_weights = [(normalized_x_corr_register_3D(overlaps[0], overlaps[1], max_shift),
        #                                        min(np.mean(np.ravel(overlaps[0])), np.mean(np.ravel(overlaps[1]))) for
        #                                       overlaps in volumes_to_register]
    else:
        pairwise_registrations_and_weights = [(x_corr_register_3D(overlaps[0], overlaps[1], max_shift),
                                               min(np.mean(np.ravel(overlaps[0])), np.mean(np.ravel(overlaps[1])))) for
                                               overlaps in volumes_to_register]

    def least_squares_traslations(pairwise_registrations_and_weights, registration_position_channel_indices):
        #Put into least squares matrix to solve for tile translations up to additive constant
        # set absolute translations for position 0 equal to zero to define absolut coordiante system
        num_positions = len(stacks)
        # specify an absolute translation of position 1 as 0,0,0 (doesn't matter bc glabal coordinates arbitrary anyway)
        A = np.zeros((3, 3 * num_positions))
        A[0, 0] = 1
        A[1, 1] = 1
        A[2, 2] = 1
        b = [0, 0, 0]
        W = [1, 1, 1]
        for i in range(len(pairwise_registrations_and_weights)):
            two_tile_registration, weight = pairwise_registrations_and_weights[i]
            pos1, pos2, channel = registration_position_channel_indices[i]
            rescaled_weight = max(1e-30, weight - backgrounds[channel_index])
            W.extend(3*[rescaled_weight])
            b.extend(two_tile_registration)
            a = np.zeros((3, 3*num_positions))
            a[0, pos2 * 3] = 1
            a[0, pos1 * 3] = -1
            a[1, pos2 * 3 + 1] = 1
            a[1, pos1 * 3 + 1] = -1
            a[2, pos2 * 3 + 2] = 1
            a[2, pos1 * 3 + 2] = -1
            A = np.concatenate((A, a), 0)
        b = np.array(b)
        w = np.sqrt(np.diag(W))
        #rewight to do weighted least sqaures
        b_w = np.dot(w, b)
        A_w = np.dot(w, A)
        # solve with least squares solver
        x = np.linalg.lstsq(A_w, b_w)[0]
        #make global translations indexed by position index
        global_translations = -np.reshape(np.round(x), ( -1, 3)).astype(np.int)
        #Use global translations to stitch together timepoint into full volume
        #gloabal_translations is in z, y,x format
        #make all z translations positive
        global_translations -= np.min(global_translations, axis=0)
        return global_translations

    ls_traslations = least_squares_traslations(pairwise_registrations_and_weights, registration_position_channel_indices)
    # zero center translation params, since offset is arbitrary
    ls_traslations[:, 1:] -= np.round((np.max(ls_traslations[:, 1:], axis=0) + np.min(ls_traslations[:, 1:], axis=0)) / 2).astype(np.int)
    #invert xy translations so they work correctly
    ls_traslations[:, 1:] *= -1
    return ls_traslations


def stitch_single_channel(stacks, translations, registrations, tile_overlap, row_col_coords, channel_index,
                          backgrounds=None):
    """
    Stitch raw stacks into single volume
    :param raw_data: dict with positions as keys containing list with 1 3d numpy array of pixels for each channel
    :param params:
    :return:
    """
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
        """
        If registration pushes this tile out of bounds, add in pixels from a neighboring tile
        :param p_index: poistion index
        :param axis: 0 or 1 for vertical or horizontal
        :param border_size: the border offset used to place the tile in its spot
        :param inserted_tile: the cropped tile to be inserted, which will be modified to have pieces of the neighbor tile
        :return:
        """
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
        if np.any(np.logical_and(row_col_coords[:, 0] == row_col[0], row_col_coords[:, 1] == row_col[1])):
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
                    axis0_neighbor_tile_coords[axis0_neighbor_tile_coords > stack_shape[1]] = stack_shape[1]
                    strip_destination = strip_destination[:axis0_neighbor_tile_coords[1] - axis0_neighbor_tile_coords[0], :]
                if axis0_neighbor_tile_coords[0] < 0:
                    axis0_neighbor_tile_coords[axis0_neighbor_tile_coords < 0] = 0
                    strip_destination = strip_destination[-(axis0_neighbor_tile_coords[1] - axis0_neighbor_tile_coords[0]):, :]
                if axis1_neighbor_tile_coords[1] > stack_shape[2]:
                    axis1_neighbor_tile_coords[axis1_neighbor_tile_coords > stack_shape[2]] = stack_shape[2]
                    strip_destination = strip_destination[:, :axis1_neighbor_tile_coords[1] - axis1_neighbor_tile_coords[0]]
                if axis1_neighbor_tile_coords[0] < 0:
                    axis1_neighbor_tile_coords[axis1_neighbor_tile_coords < 0] = 0
                    strip_destination = strip_destination[:, -(axis1_neighbor_tile_coords[1] - axis1_neighbor_tile_coords[0]):]

                #add stuff from neighboring tile into strip if theres even anything to add
                if np.ptp(axis0_neighbor_tile_coords) != 0 and np.ptp(axis1_neighbor_tile_coords) != 0:
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
