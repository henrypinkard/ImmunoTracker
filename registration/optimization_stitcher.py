import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()

from stitcher import stitch_all_channels
from scipy.optimize import minimize
import os
import scipy.ndimage as ndi

def _generate_grid(image, zyx_translations=None, yx_translations=None):
    """
    Generate Nx3 (# of pixels) x (zyx) gird of coordinates to sample new pixels
    :param image:
    :param zyx_translations: 1 x 3 global translation for the whole stack
    :param yx_translations: (# z) x 2 (y,x ) slice by slice registrations
    :return:
    """
    #get zyx coords of all points
    zyx = tf.meshgrid(*[tf.convert_to_tensor(np.arange(d), dtype=tf.int32) for d in image.shape[:3]], indexing='ij')
    #flatten into nx3 vector
    n_by_zyx = tf.cast(tf.stack([tf.reshape(d, [-1]) for d in zyx], axis=1), tf.float32)
    if zyx_translations is not None:
        #apply globabl yxz translations
        n_by_zyx = n_by_zyx + tf.cast(zyx_translations, tf.float32)
    if yx_translations is not None:
        #apply per image shifts
        #reshape to make first axis z again
        per_slice_coords = tf.reshape(n_by_zyx, [image.shape[0], -1, 3])
        #add in empty z coord
        per_slice_shifts = tf.concat([tf.zeros((yx_translations.shape[0], 1), dtype=tf.float32), tf.cast(yx_translations, tf.float32)], axis=1)
        per_slice_coords += tf.reshape(per_slice_shifts, [yx_translations.shape[0], -1, 3])
        n_by_zyx = tf.reshape(per_slice_coords, [-1, 3])
    return n_by_zyx

def _sample_pixels(image, n_by_zyx, fill_val=128):
    """
    Do bilinear sampling of all pixels with the given coordinates
    :param image: original zyx image stack
    :param n_by_zyx: n by 3 float coords of pixels
    :return: Nx1 pixel values
    """
    #get pixel values of corners
    shape_array = np.array(image.shape)
    #split to bounding integer values N x 8 x 3. the 8 is every combination of floor and ceil values
    floor_indices = tf.cast(tf.floor(n_by_zyx), tf.float32)
    ceil_indices = 1 + floor_indices
    combos = [[floor_indices[:, 0], floor_indices[:, 1], floor_indices[:, 2]],
             [floor_indices[:, 0], floor_indices[:, 1], ceil_indices[:, 2]],
             [floor_indices[:, 0], ceil_indices[:, 1], floor_indices[:, 2]],
             [floor_indices[:, 0], ceil_indices[:, 1], ceil_indices[:, 2]],
             [ceil_indices[:, 0], floor_indices[:, 1], floor_indices[:, 2]],
             [ceil_indices[:, 0], floor_indices[:, 1], ceil_indices[:, 2]],
             [ceil_indices[:, 0], ceil_indices[:, 1], floor_indices[:, 2]],
             [ceil_indices[:, 0], ceil_indices[:, 1], ceil_indices[:, 2]] ]
    corner_indices = tf.stack([tf.stack(c, axis=1) for c in combos], axis=1)
    #reshape to n x 3
    int_indices_flat = tf.reshape(corner_indices, [-1, 3])

    #clip indices so indexing works
    clipped_indices = tf.stack([tf.clip_by_value(int_indices_flat[:, i], 0, shape_array[i] - 1) for i in range(3)], axis=1)
    #make mask for replacing pixel values with default value
    out_of_bounds_mask = tf.logical_or(tf.logical_or(
                        tf.logical_or(int_indices_flat[:, 0] < 0, int_indices_flat[:, 0] > shape_array[0]),
                        tf.logical_or(int_indices_flat[:, 1] < 0, int_indices_flat[:, 1] > shape_array[1])),
                        tf.logical_or(int_indices_flat[:, 2] < 0, int_indices_flat[:, 2] > shape_array[2]))
    pix_val_list = []
    for c in ([0] if image.ndim == 3 else range(image.shape[3])):
        channel_image = image[..., c] if image.ndim == 4 else image
        pixel_vals_flat = tf.gather_nd(channel_image, tf.cast(clipped_indices, tf.int32))
        #now replace values
        pixel_vals_flat = tf.where(out_of_bounds_mask, fill_val*tf.ones_like(pixel_vals_flat), pixel_vals_flat)
        pix_val_list.append(tf.reshape(pixel_vals_flat, [-1, 8]))
    pixel_values = tf.stack(pix_val_list, axis=2)

    #calculate weights: N x 8
    ceil_weights = n_by_zyx - tf.cast(floor_indices, tf.float32)
    floor_weights = 1.0 - ceil_weights
    corner_weights = tf.stack([floor_weights[:, 0] * floor_weights[:, 1] * floor_weights[:, 2],
                                 floor_weights[:, 0] * floor_weights[:, 1] * ceil_weights[:, 2],
                                 floor_weights[:, 0] * ceil_weights[:, 1] * floor_weights[:, 2],
                                 floor_weights[:, 0] * ceil_weights[:, 1] * ceil_weights[:, 2],
                                 ceil_weights[:, 0] * floor_weights[:, 1] * floor_weights[:, 2],
                                 ceil_weights[:, 0] * floor_weights[:, 1] * ceil_weights[:, 2],
                                 ceil_weights[:, 0] * ceil_weights[:, 1] * floor_weights[:, 2],
                                 ceil_weights[:, 0] * ceil_weights[:, 1] * ceil_weights[:, 2]], axis=1)

    #add together to get pixels values
    interpolated_pixels = tf.reduce_sum(tf.cast(pixel_values, tf.float32) * corner_weights[:, :, None], axis=1)
    return tf.reshape(interpolated_pixels, image.shape)

def intra_stack_alignment_graph(yx_translations, zyxc_stack, fill_val, stack_learning_rate, stack_reg=1e-3):
    interpolated = _interpolate_stack(zyxc_stack, fill_val=fill_val, yx_translations=yx_translations)
    mean_intensity = np.mean(zyxc_stack ** 2)
    loss = tf.reduce_mean((interpolated[1:, :, :] - interpolated[:-1, :, :]) ** 2)
    loss = loss / mean_intensity
    loss = loss + stack_reg * tf.reduce_mean(yx_translations ** 2)
    optimizer = tf.train.AdamOptimizer(learning_rate=stack_learning_rate)
    optimize_op = optimizer.minimize(loss)
    return loss, optimize_op

def inter_stack_stitch_graph(p_yx_translations, p_zyx_translations_initial, p_zyxc_stacks, row_col_coords,
                             overlap_shape,  stitch_regularization_xy, stitch_regularization_z,
                             pixel_size_z, pixel_size_xy, fill_val):
    """
    Compute a loss function based on the overlap of different tiles
    """
    p_zyx_translations_flat_um = tf.placeholder(dtype=tf.float32, shape=(p_zyx_translations_initial.size,), name='p_zyx_translations_flat_um')
    p_zyx_translations_um = tf.reshape(p_zyx_translations_flat_um, [-1, 3]) #put into p_zyx shape
    p_zyx_translations = p_zyx_translations_um / tf.tile([[pixel_size_z, pixel_size_xy, pixel_size_xy]], (len(row_col_coords), 1))
    translated_stacks = [_interpolate_stack(p_zyxc_stacks[pos_index], fill_val=fill_val,
                    zyx_translations=p_zyx_translations[index] + p_zyx_translations_initial[index],
                    yx_translations=p_yx_translations[index]) for index, pos_index in enumerate(p_zyxc_stacks.keys())]

    # make sure z translations are all positive
    overlap_losses = []
    for position_index1 in range(len(translated_stacks)):
        row1, col1 = row_col_coords[position_index1]
        stack1 = translated_stacks[position_index1]
        for position_index2 in range(position_index1):
            row2, col2 = row_col_coords[position_index2]
            stack2 = translated_stacks[position_index2]
            if not ((row1 == row2 + 1 and col1 == col2) or (row1 == row2 and col1 == col2 + 1)):
                continue  # non adjacent tiles
            if row1 == row2 + 1 and col1 == col2:
                # stack1 is below stack2
                overlap1 = stack1[:, :overlap_shape[0], :]
                overlap2 = stack2[:, -overlap_shape[0]:, :]
            elif row1 == row2 and col1 == col2 + 1:
                overlap1 = stack1[:, :, :overlap_shape[1]]
                overlap2 = stack2[:, :, -overlap_shape[1]:]
            else:
                raise Exception('This shouldnt happen!')
            floverlap1 = tf.cast(overlap1, tf.float32)
            floverlap2 = tf.cast(overlap2, tf.float32)
            numer = tf.reduce_mean(floverlap1 * floverlap2) ** 2
            denom = tf.maximum(tf.convert_to_tensor(1e-10), tf.reduce_mean(floverlap1 ** 2) * tf.reduce_mean(floverlap2 ** 2))
            #multiply by number of channels so stitching when more info used puts less weight on regularization
            overlap_losses.append(numer / denom * (1.0 if len(floverlap1.shape) == 3 else floverlap1.shape[3].value))

    loss = -tf.reduce_mean(overlap_losses)

    anisotropic_penalty = tf.reshape(tf.convert_to_tensor(np.sqrt(np.tile(
        [[stitch_regularization_z, stitch_regularization_xy, stitch_regularization_xy]],
        (len(row_col_coords), 1))), tf.float32), [-1])
    translation_penalty = p_zyx_translations_flat_um * anisotropic_penalty

    loss = tf.cast(loss, tf.float32) + tf.reduce_mean(translation_penalty ** 2)
    grad = tf.gradients(loss, p_zyx_translations_flat_um)[0]
    hessian = tf.hessians(loss, p_zyx_translations_flat_um)[0]

    return loss, grad, hessian, p_zyx_translations_flat_um

def _interpolate_stack(img, fill_val, zyx_translations=None, yx_translations=None):
    """
    Bilinear resample and interpolate an image stack
    :param image: zyx or zyxc image stack
    :param fill_val: what to make the background where no data available
    :param zyx_translations: 3 element vector for glabl translation of the stack
    :param yx_translations: vector of slice by slice registration (equal to 2 * image.shape[0])
    :return: resampled image stack
    """
    if yx_translations is not None:
        yx_translations = tf.reshape(yx_translations, [-1, 2])
    grid = _generate_grid(img, yx_translations=yx_translations, zyx_translations=zyx_translations)
    if img.ndim == 4: #trailing channel dimension
        resampled = _sample_pixels(img, grid, fill_val=fill_val)
    else:
        resampled = _sample_pixels(img, grid, fill_val=fill_val)
    return resampled


def _interpolate_time_series(t_zyx_series, fill_val, t_z_translations):
    """
    Bilinear resample and interpolate an image stack
    :param image: zyx or zyxc image stack
    :param fill_val: what to make the background where no data available
    :param zyx_translations: 3 element vector for glabl translation of the stack
    :param yx_translations: vector of slice by slice registration (equal to 2 * image.shape[0])
    :return: resampled image stack
    """
    zyx = tf.meshgrid(*[tf.convert_to_tensor(np.arange(d), dtype=tf.int32) for d in img.shape[:3]], indexing='ij')
    #flatten into nx3 vector
    n_by_zyx = tf.cast(tf.stack([tf.reshape(d, [-1]) for d in zyx], axis=1), tf.float32)

    resampled = []
    for t in range(len(t_zyx_series)):
        translated_coords = n_by_zyx + tf.concat([t_z_translations[t], tf.constant([0]), tf.constant([0])])
        resampled.append(_sample_pixels(t_zyx_series[t], translated_coords, fill_val=fill_val))
    return resampled


# def export_stitched_tiff(raw_stacks, row_col_coords, overlap_shape, intra_stack_params, stitch_params, name,
#                              path='/media/hugespace/henry/lymphosight/optimization_testing/'):
#     print('stitching and exporting')
#     stitched = stitch_all_channels(raw_stacks, stitch_params, intra_stack_params, overlap_shape, row_col_coords)
#     #make channels and slices on same axis
#     stacked = np.stack(stitched, axis=0)
#     exporttiffstack(np.reshape(stacked, (stacked.shape[0] * stacked.shape[1],
#                                          stacked.shape[2], stacked.shape[3])).astype(np.uint8), path=path, name=name)
#     print('exported {}'.format(name))

def _optimize_stack(arg_list, stack_learning_rate, stack_reg):
    nonempty_pix_at_position, zyxc_stack, background, pos_index = arg_list
    print('\nOptimizing stack for position {}'.format(pos_index))
    if zyxc_stack.shape[0] == 0:
        return np.zeros((0, 2))
    elif zyxc_stack.shape[0] == 1:
        return np.zeros((1, 2))
    tf.reset_default_graph()
    yx_translations = tf.get_variable('yx_translations', [2 * sum(nonempty_pix_at_position)], initializer=tf.zeros_initializer)
    loss_op, optimize_op = intra_stack_alignment_graph(yx_translations=yx_translations, zyxc_stack=zyxc_stack,
                                                       fill_val=background, stack_learning_rate=stack_learning_rate,
                                                       stack_reg=stack_reg)
    new_min_iter = 0
    min_loss = 1e40
    iteration = 0
    min_loss_params = None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            current_yx_translations = sess.run(yx_translations)
            intra_stack_rms_shift = np.sqrt(np.mean(current_yx_translations ** 2))
            loss, h = sess.run([loss_op, optimize_op]) #run iteration
            if np.isnan(loss):
                raise Exception('NAN encounterd in loss')
            print('Stack loss: {}  \t\tstack rms: {}'.format(loss, intra_stack_rms_shift))
            # check for stopping condition
            if min_loss > loss:
                min_loss = loss
                new_min_iter = 0
                min_loss_params = current_yx_translations
            new_min_iter = new_min_iter + 1
            if new_min_iter == 10:
                break
            iteration = iteration + 1

        return min_loss_params

def _optimize_stitching(p_yx_translations, p_zyxc_stacks_stitch, p_zyx_initial, row_col_coords, overlap_shape,
                        stitch_regularization_xy, stitch_regularization_z, pixel_size_xy, pixel_size_z):
    tf.reset_default_graph()
    p_zyx_translations_flat_um = np.zeros((len(p_zyxc_stacks_stitch) * 3,))
    loss_op, grad_op, hessian_op, param_input = inter_stack_stitch_graph(
        p_yx_translations, p_zyx_initial,
        p_zyxc_stacks_stitch, row_col_coords, overlap_shape,
        stitch_regularization_xy, stitch_regularization_z, pixel_size_z, pixel_size_xy, fill_val=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def loss_fn(x):
            loss = sess.run(loss_op, feed_dict={param_input: x})
            translations = np.reshape(x, (-1, 3))
            stitch_rms_shift_z = np.sqrt(np.mean(translations[:, 0] ** 2))
            stitch_rms_shift_xy = np.mean(np.sqrt(np.sum(translations[:, 1:] ** 2, axis=1)) )
            print('Stitching loss: {}  \txy rms (um): {}  \tz rms (um): {}'.format(
                                                                loss, stitch_rms_shift_xy, stitch_rms_shift_z))
            return loss

        def grad_fn(x):
            return sess.run(grad_op, feed_dict={param_input: x})

        def hessian_fn(x):
            return sess.run(hessian_op, feed_dict={param_input: x})

        p_zyx_translations_flat_um = minimize(loss_fn, p_zyx_translations_flat_um, method='trust-exact', jac=grad_fn, hess=hessian_fn,
                       options={'gtol': 1e-3, 'disp': True, 'initial_trust_radius': 3, 'max_trust_radius': 6}).x
        #convert to pixel coords
        p_zyx_translations_pixel = np.reshape(p_zyx_translations_flat_um, [-1,3]) / np.tile(
            [[pixel_size_z, pixel_size_xy, pixel_size_xy]], (len(row_col_coords), 1))
        return p_zyx_initial + p_zyx_translations_pixel

def _optimize_z_shifts(t_zyx_img):
    tf.reset_default_graph()
    t_z_translations = tf.placeholder(dtype=tf.float32, shape=(len(t_zyx_img),), name='t_z_translations')
    interpolated = _interpolate_time_series(t_zyx_img, fill_val=0, t_z_translations=t_z_translations)
    #compute mean for normalization
    mean_intensity = np.mean(t_zyx_img ** 2)
    loss = tf.reduce_mean((interpolated[1:, :, :] - interpolated[:-1, :, :]) ** 2)
    loss = loss / mean_intensity

    #add reg if deisred
    loss_op = loss + 0.001* tf.reduce_mean(t_z_translations ** 2)
    grad_op = tf.gradients(loss, t_z_translations)[0]
    hessian_op = tf.hessians(loss, t_z_translations)[0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def loss_fn(x):
            loss = sess.run(loss_op, feed_dict={t_z_translations: x})
            translations = np.reshape(x, (-1, 3))
            stitch_rms_shift_z = np.sqrt(np.mean(translations[:, 0] ** 2))
            stitch_rms_shift_xy = np.mean(np.sqrt(np.sum(translations[:, 1:] ** 2, axis=1)) )
            print('Stitching loss: {}  \txy rms (um): {}  \tz rms (um): {}'.format(
                                                                loss, stitch_rms_shift_xy, stitch_rms_shift_z))
            return loss

        def grad_fn(x):
            return sess.run(grad_op, feed_dict={t_z_translations: x})

        def hessian_fn(x):
            return sess.run(hessian_op, feed_dict={t_z_translations: x})

        optimized_z_translations = minimize(loss_fn, t_z_translations, method='trust-exact', jac=grad_fn, hess=hessian_fn,
                       options={'gtol': 1e-4, 'disp': True, 'initial_trust_radius': 3, 'max_trust_radius': 6}).x
        return optimized_z_translations

def optimize_timepoint_stacks(p_zyxc_stacks, nonempty_pixels, intra_stack_channels,
                        stack_learning_rate=15, stack_reg=0, backgrounds=None):

    ######## optimize yx_translations for each stack
    mean_background = np.mean(backgrounds)
    arg_lists = [[np.array(nonempty_pixels[pos_index]),
                  p_zyxc_stacks[pos_index][np.array(nonempty_pixels[pos_index])][..., intra_stack_channels],
                  mean_background, pos_index]
                 for pos_index in p_zyxc_stacks.keys()]

    pos_raw_translations = []
    for a in arg_lists:
        pos_raw_translations.append(_optimize_stack(a, stack_learning_rate=stack_learning_rate, stack_reg=stack_reg))
    # zeros version for debugging
    # pos_raw_translations = [np.zeros((2 * np.sum(a[0]))) for a in arg_lists]

    # reformat and add in zeros for extra slices that weren't optimized
    p_yx_translations = []
    for index, pos_index in enumerate(list(p_zyxc_stacks.keys())):
        data_z_indices = np.where(nonempty_pixels[pos_index])[0]
        if data_z_indices.size == 0:
            first_z_index = 0
            last_z_index = -1
        else:
            first_z_index = data_z_indices[0]
            last_z_index = data_z_indices[-1]
        front_padding = np.zeros(([first_z_index, 2]), np.float32)
        optimized_translations = np.reshape(pos_raw_translations[index], [-1, 2])
        back_padding = np.zeros(([len(nonempty_pixels[pos_index]) - last_z_index - 1, 2]), np.float32)
        p_yx_translations.append(np.concatenate([front_padding, optimized_translations, back_padding], axis=0))

    return p_yx_translations


def optimize_inter_stack_stitching(p_zyxc_stacks, row_col_coords, overlap_shape, p_yx_translations,
                                   p_zyx_intitial, inter_stack_channels, pixel_size_xy, pixel_size_z,
                                   stitch_downsample_factor_xy=3, stitch_regularization_xy=0, stitch_regularization_z=0):

    if p_yx_translations is None:
        p_yx_translations = np.zeros((len(p_zyxc_stacks.keys()), 2))
    #invert xy and y so it has correct sign
    p_zyx_initial_downsampled = np.copy(p_zyx_intitial) if p_zyx_intitial is not None else np.zeros(
        [len(p_zyxc_stacks.keys()), 3])
    p_zyx_initial_downsampled[:, 1] = -p_zyx_initial_downsampled[:, 1]
    p_zyx_initial_downsampled[:, 2] = -p_zyx_initial_downsampled[:, 2]

    # means = np.mean(np.concatenate([p_zyxc_stacks[pos_index][nonempty_pixels[pos_index]] for pos_index in p_zyxc_stacks.keys()], axis=0), axis=(0, 1, 2))
    p_zyxc_stacks_stitch = {}
    # downsample, mean subtract, remove unused channels
    pixel_size_xy = pixel_size_xy * stitch_downsample_factor_xy
    for index, pos_index in enumerate(p_zyxc_stacks.keys()):
        if inter_stack_channels is None:
            stack = p_zyxc_stacks[pos_index]
        else:
            stack = p_zyxc_stacks[pos_index][..., np.array(inter_stack_channels)]

        # stack[np.logical_not(nonempty_pixels[pos_index])] = 0
        # filter and downsample
        for z in range(stack.shape[0]):
            for c in range(stack.shape[3]):
                stack[z, :, :, c] = ndi.gaussian_filter(stack[z, :, :, c], 2 * stitch_downsample_factor_xy / 6.0)
        p_zyxc_stacks_stitch[pos_index] = stack[:, ::stitch_downsample_factor_xy, ::stitch_downsample_factor_xy, :]

        p_zyx_initial_downsampled[index, 1:] = p_zyx_initial_downsampled[index, 1:] / stitch_downsample_factor_xy

        # # filter z axis
        # for channel_index, z_sigma in enumerate(stitch_z_filters):
        #     if z_sigma != -1:
        #         p_zyxc_stacks_stitch[pos_index][:, :, :, channel_index] = ndi.gaussian_filter1d(
        #             p_zyxc_stacks_stitch[pos_index][:, :, :, channel_index], sigma=z_sigma, axis=0)


    p_zyx_translations = _optimize_stitching(p_yx_translations, p_zyxc_stacks_stitch, p_zyx_initial_downsampled,
                                             row_col_coords, overlap_shape // stitch_downsample_factor_xy,
                                             stitch_regularization_xy, stitch_regularization_z,
                                             pixel_size_xy, pixel_size_z)

    # flip sign as stitcher expects
    p_zyx_translations[:, 1] = -p_zyx_translations[:, 1]
    p_zyx_translations[:, 2] = -p_zyx_translations[:, 2]

    # Rescale these translations to account for downsampling
    p_zyx_translations[:, 1:] = stitch_downsample_factor_xy * p_zyx_translations[:, 1:]
    return p_zyx_translations

def optimize_z_over_time(t_zyx_volumes, stitch_downsample_factor_xy=8):

    t_zyx_resampled = []
    for volume in t_zyx_volumes:
        # filter and downsample
        for z in range(volume.shape[0]):
            t_zyx_volumes[z, :, :] = ndi.gaussian_filter(volume[z, :, :], 2 * stitch_downsample_factor_xy / 6.0)
        t_zyx_resampled.append(volume[:, ::stitch_downsample_factor_xy, ::stitch_downsample_factor_xy])

        # # filter z axis
        # for channel_index, z_sigma in enumerate(stitch_z_filters):
        #     if z_sigma != -1:
        #         p_zyxc_stacks_stitch[pos_index][:, :, :, channel_index] = ndi.gaussian_filter1d(
        #             p_zyxc_stacks_stitch[pos_index][:, :, :, channel_index], sigma=z_sigma, axis=0)


    t_z_translations = _optimize_z_shifts(np.stack(t_zyx_resampled))

    return t_z_translations
