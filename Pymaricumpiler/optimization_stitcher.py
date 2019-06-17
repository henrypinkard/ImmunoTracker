import jax.numpy as np
from jax import jit
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from stitcher import stitch_all_channels
from PIL import Image
from transform_resampling import interpolate_stack


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

def convert_stack_params(intra_stack_params_tensor, nonempty_pixels):
    """
    Convert params used in optimization to the format expected by stitching algorithms
    """
    intra_params = [np.reshape(p.numpy(), [-1, 2]) for p in intra_stack_params_tensor]
    # add on trailing and leading zeros corresponding to nonempty pixels, and flip x and y
    full_intra_params = []
    for pos_index, stack_params in enumerate(intra_params):
        full_intra_params.append(
            np.flip(np.concatenate([np.zeros(([np.where(nonempty_pixels[pos_index])[0][0], 2]), np.float32),
                                    stack_params, np.zeros(([len(nonempty_pixels[pos_index]) -
                                                             np.where(nonempty_pixels[pos_index])[0][-1] - 1, 2]),
                                                           np.float32)], axis=0), axis=1))
    return full_intra_params

def convert_stitch_params(stitching_params_tensor):
        # compile all params in correct format for stitching--Calculated stitch params should be negated relative
        # to the ones optimized
        stitch_params = np.concatenate([stitching_params_tensor[-1].numpy()[:, None],
                                        np.concatenate([p.numpy() for p in stitching_params_tensor[:-1]], axis=0)], axis=1)
        stitch_params = np.concatenate([-stitch_params[:, 0, None], stitch_params[:, 1:]], axis=1)
        return stitch_params


def inter_stack_stitch_loss(registered_stacks, row_col_coords, overlap_shape, use_channels):
    """
    Compute a loss function based on the overlap of different tiles
    """
    # make sure z translations are all positive
    loss = np.zeros((), dtype=np.float32)
    for position_index1 in range(len(registered_stacks)):
        row1, col1 = row_col_coords[position_index1]
        stack1 = registered_stacks[position_index1]
        for position_index2 in range(position_index1):
            row2, col2 = row_col_coords[position_index2]
            stack2 = registered_stacks[position_index2]
            for channel_index in use_channels:
                if not ((row1 == row2 + 1 and col1 == col2) or (row1 == row2 and col1 == col2 + 1)):
                    continue  # non adjacent tiles
                stack1_channel = stack1[:, :, :, channel_index]
                stack2_channel = stack2[:, :, :, channel_index]
                if row1 == row2 + 1 and col1 == col2:
                    # stack1 is below stack2
                    overlap1 = stack1_channel[:, :overlap_shape[0], :]
                    overlap2 = stack2_channel[:, -overlap_shape[0]:, :]
                elif row1 == row2 and col1 == col2 + 1:
                    overlap1 = stack1_channel[:, :, :overlap_shape[1]]
                    overlap2 = stack2_channel[:, :, -overlap_shape[1]:]
                # o1_mean_sub = overlap1 - np.reduce_mean(overlap1)
                # o2_mean_sub = overlap2 - np.reduce_mean(overlap2)
                # numer = np.reduce_sum(o1_mean_sub * o2_mean_sub) ** 2
                # denom = np.reduce_sum(o1_mean_sub ** 2) * np.reduce_sum(o2_mean_sub ** 2)
                numer = np.mean(overlap1 * overlap2) ** 2
                denom = np.mean(overlap1 ** 2) * np.mean(overlap2 ** 2)
                loss += numer / denom
    return -loss

def intra_stack_alignment_loss(zyxc_stack, intra_stack_channels):
    loss = np.array([0.0])
    for channel in intra_stack_channels:
        loss += np.mean((zyxc_stack[1:, :, :, channel] - zyxc_stack[:-1, :, :, channel]) ** 2)

def optimize_timepoint(p_zyxc_stacks, nonempty_pixels, row_col_coords, overlap_shape, intra_stack_channels,
                       inter_stack_channels, pixel_size_xy, pixel_size_z, stack_learning_rate=0.3, stitch_learning_rate=0.02,
                       stitch_regularization=1e-16, stack_regularization=0.01, name='image',
                       optimization_log_dir='.'):

    def compute_stack_loss(zyxc_stack, slice_translations):
        loss = np.array([0.0])
        for channel in intra_stack_channels:
            interpolated = interpolate_stack(zyxc_stack[..., channel], fill_val=127, yx_translations=slice_translations)
            loss += intra_stack_alignment_loss(interpolated, intra_stack_channels)
        return loss

    for pos_index in p_zyxc_stacks.keys():
        slice_translations = np.zeros(2 * np.sum(nonempty_pixels[pos_index]))
        zyxc_stack = p_zyxc_stacks[pos_index]
        loss_fn = lambda t: compute_stack_loss(zyxc_stack, t)
        grad_fn = jit(grad(loss_fn))

        while True: #optimzation loop
            loss = loss_fn(slice_translations)
            grad = grad_fn(slice_translations)
            slice_translations = slice_translations - stack_learning_rate * grad
            print(loss)



    # intra_stack_rms_shift = np.sqrt(
    #     np.mean(np.concatenate([np.ravel(g.numpy()) for g in intra_stack_params_tensor]) ** 2))
    # out = 'Stack loss,stack rms,{},{}'.format(stack_loss.numpy(), intra_stack_rms_shift)

    #intialize as zeros
    # translation_params = np.zeros((len(p_zyxc_stacks) * 3, 1))
    #
    # def stitch_loss(translation_params):
    #     #TODO: add in registrations
    #     #reformat
    #     translation_params = np.reshape(translation_params, [-1, 3])
    #     #make min z translation 0
    #     translation_params -= np.min(translation_params[:, 0])
    #     registered_stacks = []
    #     for pos_index, zyxc_stack in enumerate(p_zyxc_stacks.values()):
    #         # stack = interpolate_stack_with_z_shift(translation_params[pos_index, 0], zyxc_stack, translation_params[:, 0])
    #         # stack = interpolate_stack_with_xy_shift(translation_params[pos_index, 1:], stack)
    #         # registered_stacks.append(stack)
    #         registered_stacks.append(zyxc_stack)
    #     return compute_overlap(registered_stacks, row_col_coords, overlap_shape, inter_stack_channels)



        # compute_overlap(registered_stacks, row_col_coords, overlap_shape, use_channels, all_z_translations)

        # np.sum()


        # stitch_layer = ImageStitchingLayer(row_col_coords, zyxc_stacks[0].shape, overlap_shape, inter_stack_channels)
        # full_model = tf.keras.Sequential([stacks_layer, stitch_layer])
        # stitch_optimizer = tf.train.MomentumOptimizer(learning_rate=stitch_learning_rate, momentum=0.99)
        # stitch_loss_rescale = None

    #     # Stitch optimization loop
    #     new_min_iter = 0
    #     min_loss = 10
    #     iteration = 0
    #     while True:
    #         with tf.GradientTape() as full_tape:
    #             stitching_loss = full_model(None)
    #             #get stitching params
    #             stitching_params_tensor = [all_params[2 * i + 1] for i in range(len(stacks))] + [full_model.trainable_variables[-1]]
    #             if stitch_loss_rescale is None:
    #                 stitch_loss_rescale = np.abs(stitching_loss.numpy())
    #             stitching_loss = stitching_loss / stitch_loss_rescale
    #             #add regualrization
    #             stitch_params_single_tensor = tf.concat(
    #                 [tf.concat((stitching_params_tensor[i], tf.reshape(stitching_params_tensor[-1][i], (1, 1))), axis=1)
    #                  for i in range(len(stitching_params_tensor) - 1)], axis=0)
    #             #account for anisotrpy in pixels sizes
    #             isotropic_params = tf.concat([stitch_params_single_tensor[:, :2],
    #                        tf.reshape(pixel_size_z / pixel_size_xy * stitch_params_single_tensor[:, 2], [-1, 1])], axis=1)
    #
    #             stitch_penalty = tf.reduce_mean(isotropic_params ** 2)
    #             stitching_loss = stitching_loss + stitch_regularization * stitch_penalty
    #
    #         #compute rms pixel shifts to monitor progress
    #         stitch_rms_shift = np.sqrt(np.mean(np.concatenate([np.ravel(g.numpy()) for g in stitching_params_tensor]) ** 2))
    #         out = 'stitch loss, stitch rms,{},{}'.format(stitching_loss.numpy(),  stitch_rms_shift)
    #         print(out)
    #         file.write(out + '\n')
    #
    #         #calc gradients and take a step
    #         stitch_grads = full_tape.gradient(stitching_loss, stitching_params_tensor)
    #         stitch_optimizer.apply_gradients(zip(stitch_grads, stitching_params_tensor), global_step=tf.train.get_or_create_global_step())
    #
    #         #make the mean xy shift for stitching 0
    #         mean_shift = tf.reduce_mean(tf.concat(stitching_params_tensor[:-1], axis=0), axis=0)
    #         for xy_shift in stitching_params_tensor[:-1]:
    #             xy_shift.assign(xy_shift - mean_shift)
    #
    #         #check for stopping condition
    #         if min_loss > stitching_loss.numpy():
    #             min_loss = stitching_loss.numpy()
    #             new_min_iter = 0
    #         new_min_iter = new_min_iter + 1
    #         if new_min_iter == 10:
    #             break
    #         iteration = iteration + 1

    stitch_params = None
    # stitch_params = convert_stitch_params(stitching_params_tensor)
    # return stack_params, stitch_params
    #TODO
    return None
