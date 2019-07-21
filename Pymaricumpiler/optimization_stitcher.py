import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()

from stitcher import stitch_all_channels
from PIL import Image
import os
import scipy.ndimage as ndi
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def _generate_grid(image, zyx_translations=None, yx_translations=None):
    """
    Generate Nx3 (# of pixels) x (zyx) gird of coordinates to sample new pixels
    :param image:
    :param zyx_translations: 1 x 3 global translation for the whole stack
    :param yx_translations: (# z) x 2 (y,x ) slice by slice registrations
    :return:
    """
    #get zyx coords of all points
    zyx = tf.meshgrid(*[tf.convert_to_tensor(np.arange(d)) for d in image.shape[:3]], indexing='ij')
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
        per_slice_shifts = tf.concat([tf.zeros((yx_translations.shape[0], 1)), tf.cast(yx_translations, tf.float32)], axis=1)
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
    floor_indices = tf.cast(tf.floor(n_by_zyx), tf.int32)
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
        pixel_vals_flat = tf.gather_nd(channel_image, clipped_indices)
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

def inter_stack_stitch_graph(p_yx_translations, p_zyx_translations_flat, p_zyxc_stacks, row_col_coords, overlap_shape, fill_val, stitch_regularizaton=0.0):
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
            numer = tf.reduce_mean(overlap1 * overlap2) ** 2
            denom = tf.reduce_mean(overlap1 ** 2) * tf.reduce_mean(overlap2 ** 2)
            loss += numer / denom

    loss = -loss
    #TODO: add anisotropic regularization
    loss = loss + stitch_regularizaton * tf.reduce_mean(p_zyx_translations_flat ** 2)

    grad = tf.gradients(loss, p_zyx_translations_flat)[0]
    hessian = tf.hessians(loss, p_zyx_translations_flat)[0]

    newton_delta = tf.placeholder(tf.float32, p_zyx_translations_flat.shape)
    assign_op = tf.assign_sub(p_zyx_translations_flat, newton_delta)

    return loss, grad, hessian, newton_delta, assign_op

    # hess_grad_prod = tf.matmul(inv_hess, tf.reshape(grad, [-1, 1]))
    # newton_delta = tf.reshape(hess_grad_prod, [-1])
    # optimize_step = p_zyx_translations_flat.assign(newton_delta + p_zyx_translations_flat)
    # return loss, grad, hessian, inv_hess, hess_grad_prod, newton_delta, optimize_step
    # return optimize_step, loss

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

def exporttiffstack(datacube, path, name='export'):
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
    path = "{}{}.tif".format(path, name)
    imlist[0].save(path, compression="tiff_deflate", save_all=True, append_images=imlist[1:])

def export_stitched_tiff(raw_stacks, row_col_coords, overlap_shape, intra_stack_params, stitch_params, name,
                             path='/media/hugespace/henry/lymphosight/optimization_testing/'):
    print('stitching and exporting')
    stitched = stitch_all_channels(raw_stacks, stitch_params, intra_stack_params, overlap_shape, row_col_coords)
    #make channels and slices on same axis
    stacked = np.stack(stitched, axis=0)
    exporttiffstack(np.reshape(stacked, (stacked.shape[0] * stacked.shape[1],
                                         stacked.shape[2], stacked.shape[3])).astype(np.uint8), path=path, name=name)
    print('exported {}'.format(name))

def optimize_stack(arg_list):
    nonempty_pix_at_position, zyxc_stack, background = arg_list
    tf.reset_default_graph()
    yx_translations = tf.get_variable('yx_translations', [2 * sum(nonempty_pix_at_position)])
    loss_op, optimize_op = intra_stack_alignment_graph(yx_translations=yx_translations, zyx_stack=zyxc_stack, fill_val=background)
    new_min_iter = 0
    min_loss = 1e40
    iteration = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            loss, h = sess.run([loss_op, optimize_op]) #run iteration
            #print
            intra_stack_rms_shift = np.sqrt(np.mean(sess.run(yx_translations)) ** 2)
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

def optimize_stitching(p_yx_translations, p_zyx_translations, p_zyxc_stacks_stitch, row_col_coords, overlap_shape):
    loss_op, grad_op, hessian_op, newton_delta_op, assign_op = inter_stack_stitch_graph(
        p_yx_translations, p_zyx_translations,
        p_zyxc_stacks_stitch, row_col_coords, overlap_shape, fill_val=0)
    new_min_iter = 0
    min_loss = 1e40
    iteration = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            loss, grad = sess.run([loss_op, grad_op])
            hessian = sess.run([hessian_op])
            newton_delta = np.dot(np.linalg.inv(hessian), grad)
            sess.run([assign_op], feed_dict={newton_delta_op: np.ravel(newton_delta)})
            stitch_rms_shift = np.sqrt(np.mean(sess.run(p_zyx_translations)) ** 2)
            print('Stitching loss: {}  \t\tstitch rms: {}'.format(loss, stitch_rms_shift))
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
                       inter_stack_channels, pixel_size_xy, pixel_size_z, stitch_learning_rate=0.02, downsample_factor=2,
                       stitch_regularization=1e-16, name='image',
                       optimization_log_dir='./', backgrounds=None, swap_xy=False, invert_xy=False):
    with np.load('{}{}__yx_translations.npz'.format(optimization_log_dir, name)) as loaded:
        p_yx_translations = loaded['p_yx_translations']
        p_zyx_translations = loaded['p_zyx_translations']

    if invert_xy:
        p_zyx_translations[:, 1] = -p_zyx_translations[:, 1]
        p_zyx_translations[:, 2] = -p_zyx_translations[:, 2] 

    if swap_xy:
            tmp1 = p_zyx_translations[:, 1]
            tmp2 = p_zyx_translations[:, 2]
            p_zyx_translations[:, 1] = tmp2
            p_zyx_translations[:, 2] = tmp1


    return p_yx_translations, p_zyx_translations



    # ######## optimize yx_translations for each stack
    # mean_background = np.mean(backgrounds)
    # arg_lists = [[np.array(nonempty_pixels[pos_index]), p_zyxc_stacks[pos_index][np.array(nonempty_pixels[
    #                     pos_index])][..., intra_stack_channels], mean_background] for pos_index in p_zyxc_stacks.keys()]

    # # with Pool(6) as p:
    # #     pos_raw_translations = p.map(optimize_stack, arg_lists)
    # pos_raw_translations = [optimize_stack(a) for a in arg_lists]
    # #zeros version for debugging
    # # pos_raw_translations = [np.zeros((2 * np.sum(a[0]))) for a in arg_lists]
    
    # #reformat and add in zeros for extra slices that weren't optimized
    # p_yx_translations = [np.concatenate([np.zeros(([np.where(nonempty_pixels[pos_index])[0][0], 2]), np.float32),
    #             np.reshape(pos_raw_translations[pos_index], [-1, 2]), np.zeros(([len(nonempty_pixels[pos_index]) -
    #             np.where(nonempty_pixels[pos_index])[0][-1] - 1, 2]), np.float32)], axis=0)
    #                      for pos_index in p_zyxc_stacks.keys()]
    
    # p_yx_translations = np.stack(p_yx_translations, axis=0)
    


    # ########## Now optimizing stitching
    # # means = np.mean(np.concatenate([p_zyxc_stacks[pos_index][nonempty_pixels[pos_index]] for pos_index in p_zyxc_stacks.keys()], axis=0), axis=(0, 1, 2))
    # p_zyxc_stacks_stitch = {}
    # #downsample, mean subtract, remove unused channels
    # for pos_index in p_zyxc_stacks.keys():
    #     # stack = p_zyxc_stacks[pos_index][..., np.array(inter_stack_channels)] - means[None, None, None, np.array(inter_stack_channels)]
    #     stack = p_zyxc_stacks[pos_index][..., np.array(inter_stack_channels)]

    #     # stack[np.logical_not(nonempty_pixels[pos_index])] = 0
    #     #filter and downsample
    #     for z in np.where(np.array(nonempty_pixels[pos_index]))[0]:
    #         for c in range(stack.shape[3]):
    #             stack[z, :, :, c] = ndi.gaussian_filter(stack[z, :, :, c], 2*downsample_factor / 6.0)
    #     p_zyxc_stacks_stitch[pos_index] = stack[:, ::downsample_factor, ::downsample_factor, :]
    # #TODO: add in anisotropic regularization?
    
    # tf.reset_default_graph()
    # p_zyx_translations = tf.get_variable('p_zyx_translations', len(p_zyxc_stacks) * 3)
    # p_zyx_translations_optimized = optimize_stitching(p_yx_translations, p_zyx_translations, p_zyxc_stacks_stitch,
    #         row_col_coords, overlap_shape // downsample_factor)
    
    # p_zyx_translations = np.reshape(p_zyx_translations_optimized, [-1, 3])
    
    # #Rescale these translations to account for downsampling
    # p_zyx_translations[:, 1:] = downsample_factor * p_zyx_translations[:, 1:]
    # #TODO: more optimization at full resolution or is this good?
    
    # #TODO: check that these params are right signs etc
    
    # np.savez('{}{}__yx_translations'.format(optimization_log_dir, name),
    #         p_yx_translations=p_yx_translations, p_zyx_translations=p_zyx_translations)
    
    # return p_yx_translations, p_zyx_translations
