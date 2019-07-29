import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()

# from PIL import Image
from stitcher import stitch_all_channels
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

def intra_stack_alignment_graph(yx_translations, zyx_stack, fill_val, stack_learning_rate=2, stack_regularization=1e-2):
    interpolated = _interpolate_stack(zyx_stack, fill_val=fill_val, yx_translations=yx_translations)
    loss = tf.reduce_mean((interpolated[1:, :, :] - interpolated[:-1, :, :]) ** 2)
    loss = loss + stack_regularization * tf.reduce_mean(yx_translations ** 2)
    optimizer = tf.train.AdamOptimizer(learning_rate=stack_learning_rate)
    optimize_op = optimizer.minimize(loss)
    return loss, optimize_op

def inter_stack_stitch_graph(p_yx_translations, p_zyx_translations_flat, p_zyxc_stacks, row_col_coords,
                             overlap_shape,  stitch_regularization, z_to_xy_ratio, fill_val):
    """
    Compute a loss function based on the overlap of different tiles
    """

    p_zyx_translations = tf.reshape(p_zyx_translations_flat, [-1, 3])
    translated_stacks = {pos_index: _interpolate_stack(p_zyxc_stacks[pos_index], fill_val=fill_val,
                    zyx_translations=p_zyx_translations[pos_index], yx_translations=p_yx_translations[pos_index])
                         for pos_index in p_zyxc_stacks.keys()}

    # make sure z translations are all positive
    loss = 0.0
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
            denom = tf.reduce_mean(floverlap1 ** 2) * tf.reduce_mean(floverlap2 ** 2)
            loss += numer / denom

    loss = -loss
    pixel_size_rescale = tf.reshape(tf.convert_to_tensor(np.tile([[
        1.0, 1.0 / z_to_xy_ratio, 1.0 / z_to_xy_ratio]], (len(row_col_coords), 1)), tf.float32), [-1])
    rescaled_translations = pixel_size_rescale * p_zyx_translations_flat
    loss = tf.cast(loss, tf.float32) + stitch_regularization * tf.reduce_mean(rescaled_translations ** 2)
    grad = tf.gradients(loss, p_zyx_translations_flat)[0]
    hessian = tf.hessians(loss, p_zyx_translations_flat)[0]

    newton_delta = tf.placeholder(tf.float32, p_zyx_translations_flat.shape)
    assign_op = tf.assign_sub(p_zyx_translations_flat, newton_delta)

    return loss, grad, hessian, newton_delta, assign_op

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

# def export_stitched_tiff(raw_stacks, row_col_coords, overlap_shape, intra_stack_params, stitch_params, name,
#                              path='/media/hugespace/henry/lymphosight/optimization_testing/'):
#     print('stitching and exporting')
#     stitched = stitch_all_channels(raw_stacks, stitch_params, intra_stack_params, overlap_shape, row_col_coords)
#     #make channels and slices on same axis
#     stacked = np.stack(stitched, axis=0)
#     exporttiffstack(np.reshape(stacked, (stacked.shape[0] * stacked.shape[1],
#                                          stacked.shape[2], stacked.shape[3])).astype(np.uint8), path=path, name=name)
#     print('exported {}'.format(name))

def optimize_stack(arg_list):
    nonempty_pix_at_position, zyxc_stack, background = arg_list
    if zyxc_stack.shape[0] == 0:
        return np.zeros((0, 2))
    elif zyxc_stack.shape[0] == 1:
        return np.zeros((1, 2))
    tf.reset_default_graph()
    yx_translations = tf.get_variable('yx_translations', [2 * sum(nonempty_pix_at_position)], initializer=tf.zeros_initializer)
    loss_op, optimize_op = intra_stack_alignment_graph(yx_translations=yx_translations, zyx_stack=zyxc_stack, fill_val=background)
    new_min_iter = 0
    min_loss = 1e40
    iteration = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            intra_stack_rms_shift = np.sqrt(np.mean(sess.run(yx_translations)) ** 2)
            loss, h = sess.run([loss_op, optimize_op]) #run iteration
            if np.isnan(loss):
                raise Exception('NAN encounterd in loss')
            #TODO:
            print('breaking early')
            break
            #print
            print('Stack loss: {}  \t\tstack rms: {}'.format(loss, intra_stack_rms_shift))
            # check for stopping condition
            if min_loss > loss:
                min_loss = loss
                new_min_iter = 0
            new_min_iter = new_min_iter + 1
            if new_min_iter == 10:
                break
            iteration = iteration + 1

        return sess.run(yx_translations)

def optimize_stitching(p_yx_translations, p_zyx_translations, p_zyxc_stacks_stitch, row_col_coords, overlap_shape,
                       stitch_regularization, pixel_size_xy, pixel_size_z):
    z_to_xy_ratio = pixel_size_z / pixel_size_xy
    loss_op, grad_op, hessian_op, newton_delta_op, assign_op = inter_stack_stitch_graph(
        p_yx_translations, p_zyx_translations,
        p_zyxc_stacks_stitch, row_col_coords, overlap_shape,  stitch_regularization, z_to_xy_ratio, fill_val=0)
    new_min_iter = 0
    min_loss = 1e40
    iteration = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            loss, grad = sess.run([loss_op, grad_op])
            if np.isnan(loss):
                raise Exception('NAN encounterd in loss')
            hessian = sess.run([hessian_op])
            print('Eigenvalues')
            print(np.linalg.eigvals(hessian))
            translations = np.reshape(sess.run(p_zyx_translations), (-1, 3))
            translations[:, 0] = translations[:, 0] * pixel_size_z
            translations[:, 1:] = translations[:, 1:] * pixel_size_xy  
            stitch_rms_shift_z = np.sqrt(np.mean(translations[:, 0] ** 2))
            stitch_rms_shift_xy = np.sqrt(np.mean(translations[:, 1:] ** 2)) 
            print('Stitching loss: {}  \txy rms (um): {}  \tz rms (um): {}'.format(loss, stitch_rms_shift_xy, stitch_rms_shift_z))
            newton_delta = np.dot(np.linalg.inv(hessian), grad)
            sess.run([assign_op], feed_dict={newton_delta_op: np.ravel(newton_delta)})
            # check for stopping condition
            if min_loss > loss:
                min_loss = loss
                new_min_iter = 0
            new_min_iter = new_min_iter + 1
            if new_min_iter == 5:
                break
            iteration = iteration + 1
        return sess.run(p_zyx_translations)

def optimize_timepoint(p_zyxc_stacks, nonempty_pixels, row_col_coords, overlap_shape, intra_stack_channels,
                       inter_stack_channels, pixel_size_xy, pixel_size_z,
                       downsample_factor=3, param_cache_dir=None,
                       stitch_regularization=1e-2, param_cache_name='.', backgrounds=None,
                       stack=True, stitch=True):
    optimized_params = {}

    saved_name = '{}{}_optimized_params.npz'.format(param_cache_dir, param_cache_name)
    if os.path.isfile(saved_name):
        with np.load(saved_name) as loaded:
            if 'p_yx_translations' in loaded and not stack:
                optimized_params['p_yx_translations'] = loaded['p_yx_translations']
            if 'p_zyx_translations' in loaded and not stitch: 
                optimized_params['p_zyx_translations'] = loaded['p_zyx_translations']

    if stack:
        ######## optimize yx_translations for each stack
        mean_background = np.mean(backgrounds)
        arg_lists = [[np.array(nonempty_pixels[pos_index]),
                       p_zyxc_stacks[pos_index][np.array(nonempty_pixels[pos_index])][..., intra_stack_channels],
                       mean_background]
                      for pos_index in p_zyxc_stacks.keys()]
        
        pos_raw_translations = [optimize_stack(a) for a in arg_lists]
        #zeros version for debugging
        # pos_raw_translations = [np.zeros((2 * np.sum(a[0]))) for a in arg_lists]
        
        #reformat and add in zeros for extra slices that weren't optimized
        p_yx_translations = []
        for pos_index in p_zyxc_stacks.keys():
            data_z_indices = np.where(nonempty_pixels[pos_index])[0]
            if data_z_indices.size == 0:
                first_z_index = 0
                last_z_index = -1
            else:
                first_z_index = data_z_indices[0]
                last_z_index = data_z_indices[-1]
            front_padding = np.zeros(([first_z_index, 2]), np.float32)
            optimized_translations = np.reshape(pos_raw_translations[pos_index], [-1, 2])
            back_padding = np.zeros(([len(nonempty_pixels[pos_index]) - last_z_index - 1, 2]), np.float32)
            p_yx_translations.append(np.concatenate([front_padding, optimized_translations, back_padding], axis=0))

        optimized_params['p_yx_translations'] = np.stack(p_yx_translations, axis=0)
    

    if stitch:
        ########## Now optimizing stitching
        # means = np.mean(np.concatenate([p_zyxc_stacks[pos_index][nonempty_pixels[pos_index]] for pos_index in p_zyxc_stacks.keys()], axis=0), axis=(0, 1, 2))
        p_zyxc_stacks_stitch = {}
        #downsample, mean subtract, remove unused channels
        pixel_size_xy = pixel_size_xy * downsample_factor
        for pos_index in p_zyxc_stacks.keys():
            # stack = p_zyxc_stacks[pos_index][..., np.array(inter_stack_channels)] - means[None, None, None, np.array(inter_stack_channels)]
            stack = p_zyxc_stacks[pos_index][..., np.array(inter_stack_channels)]

            # stack[np.logical_not(nonempty_pixels[pos_index])] = 0
            #filter and downsample
            for z in np.where(np.array(nonempty_pixels[pos_index]))[0]:
                for c in range(stack.shape[3]):
                    stack[z, :, :, c] = ndi.gaussian_filter(stack[z, :, :, c], 2*downsample_factor / 6.0)
            p_zyxc_stacks_stitch[pos_index] = stack[:, ::downsample_factor, ::downsample_factor, :]

        tf.reset_default_graph()
        p_zyx_translations = tf.get_variable('p_zyx_translations', len(p_zyxc_stacks) * 3, initializer=tf.zeros_initializer)
        p_zyx_translations_optimized = optimize_stitching(p_yx_translations, p_zyx_translations, p_zyxc_stacks_stitch,
                row_col_coords, overlap_shape // downsample_factor, stitch_regularization, pixel_size_xy, pixel_size_z)
        
        p_zyx_translations = np.reshape(p_zyx_translations_optimized, [-1, 3])
        #flip sign as stitcher expects
        p_zyx_translations[:, 1] = -p_zyx_translations[:, 1]
        p_zyx_translations[:, 2] = -p_zyx_translations[:, 2]

        #Rescale these translations to account for downsampling
        p_zyx_translations[:, 1:] = downsample_factor * p_zyx_translations[:, 1:]
        optimized_params['p_zyx_translations'] = p_zyx_translations


    np.savez('{}{}_optimized_params'.format(param_cache_dir, param_cache_name), **optimized_params)
    
    return optimized_params
