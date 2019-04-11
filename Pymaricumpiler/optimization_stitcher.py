import tensorflow as tf
import numpy as np
from transformer import spatial_transformer_network
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from stitcher import stitch_all_channels
from PIL import Image
from scipy.ndimage import filters


class IndividualStacksLayer(tf.keras.layers.Layer):
  #batch size is the number of slices in the z stack
  def __init__(self, zyxc_stacks, nonempty_pixels):
      super(IndividualStacksLayer, self).__init__()
      self.images = []
      self.nums_z_slices = []
      self.nonempty_pixel_masks =[]
      for pos_index in range(len(zyxc_stacks)):
          self.images.append(tf.convert_to_tensor(zyxc_stacks[pos_index][nonempty_pixels[pos_index]].astype(np.float32)))
          self.nums_z_slices.append(np.sum(nonempty_pixels[pos_index]))
          self.nonempty_pixel_masks.append(nonempty_pixels[pos_index])
      # compute averge intensity at each slice to get normalization
      self.height = self.images[0].shape[1].value
      self.width = self.images[0].shape[2].value

  def build(self, input_shape):
    # The build method gets called the first time your layer is used.
    # Creating variables on build() allows you to make their shape depend
    # on the input shape and hence removes the need for the user to specify
    # full shapes. It is possible to create variables during __init__() if
    # you already know their full shapes.
    self.translation_weights = []
    self.stack_translation_weights = []
    for pos_index in range(len(self.images)):
        self.translation_weights.append(self.add_variable("slice_translations", [self.nums_z_slices[pos_index] * 2], dtype=tf.float32,
                                                  initializer=tf.keras.initializers.zeros))
        self.stack_translation_weights.append(self.add_variable("stack_translations_xy", [1, 2], dtype=tf.float32,
                                                     initializer=tf.keras.initializers.zeros))

  def call(self, input):
      full_stacks =[]
      for pos_index in range(len(self.images)):
          abs_translations = tf.reshape(self.translation_weights[pos_index], [self.nums_z_slices[pos_index], 2])
          abs_translations += self.stack_translation_weights[pos_index]
          identities = tf.eye(2, 2, [abs_translations.shape[0].value])
          #normalizing so translations can be kept in pixel coordinates
          normalized_translations = abs_translations / np.array([self.height / 2, self.width / 2])[None, :]
          affines = tf.concat([identities, normalized_translations[:, :, None]], axis=2)
          transformed = spatial_transformer_network(self.images[pos_index], tf.reshape(affines, [-1, 6]))
          #add back in the slices where no data collected
          trailing_dims = [d.value for d in transformed.shape[1:]]
          full_stacks.append(tf.concat([tf.zeros(([np.where(self.nonempty_pixel_masks[pos_index])[0][0], *trailing_dims]), tf.float32),
                     transformed,
                     tf.zeros(([len(self.nonempty_pixel_masks[pos_index]) -
                                np.where(self.nonempty_pixel_masks[pos_index])[0][-1] - 1, *trailing_dims]), tf.float32)
                     ], axis=0))
      return full_stacks

class ImageStitchingLayer(tf.keras.layers.Layer):

  def __init__(self, row_col_coords, stack_shape, overlap_shape, use_channels):
      super(ImageStitchingLayer, self).__init__()
      self.num_positions = row_col_coords.shape[0]
      self.row_col_coords = row_col_coords
      self.stack_shape = stack_shape
      self.overlap_shape = overlap_shape
      self.use_channels = use_channels

  def build(self, input_shape):
    # The build method gets called the first time your layer is used.
    # Creating variables on build() allows you to make their shape depend
    # on the input shape and hence removes the need for the user to specify
    # full shapes. It is possible to create variables during __init__() if
    # you already know their full shapes.
    self.z_translation_weights = self.add_variable("stack_translations_z", [self.num_positions], dtype=tf.float32,
                                              initializer=tf.keras.initializers.zeros)

  def _interpolate_stack(self, z_shift, stack):
      """
      :param z_shift:
      :param stack:
      :return:
      """
      min_z_index = tf.cast(tf.floor(tf.reduce_min(self.z_translation_weights)), tf.int32)
      max_z_index = tf.cast(tf.ceil(tf.reduce_max(self.z_translation_weights)), tf.int32) + self.stack_shape[0]
      stack_z_shape = max_z_index - min_z_index
      # generate interpolated stack based on float z shift
      stack_z_offset = tf.cast(tf.floor(z_shift), tf.int32)
      top_stack = tf.concat([tf.zeros((stack_z_offset, self.stack_shape[1], self.stack_shape[2])), stack,
                             tf.zeros(((stack_z_shape + 1) - (stack_z_offset + stack.shape[0]), self.stack_shape[1], self.stack_shape[2]))], axis=0)
      bottom_stack = tf.concat([tf.zeros((stack_z_offset + 1, self.stack_shape[1], self.stack_shape[2])), stack,
                                tf.zeros((stack_z_shape - (stack_z_offset + stack.shape[0]), self.stack_shape[1],
                                          self.stack_shape[2]))], axis=0)
      bottom_weight = z_shift - tf.cast(stack_z_offset, tf.float32)
      top_weight = 1 - bottom_weight
      interpolated_stack = top_stack * top_weight + bottom_stack * bottom_weight
      return interpolated_stack

  def call(self, input):
      """
      :param input: nested a list of images indexed by position index then by channel
      :return:
      """
      #make sure z translations are all positive
      min_z_trans = tf.reduce_min(self.z_translation_weights)
      loss = tf.zeros((), dtype=tf.float32)
      for position_index1 in range(len(input)):
          row1, col1 = self.row_col_coords[position_index1]
          stack1 = input[position_index1]
          z_shift_1 = self.z_translation_weights[position_index1]
          for position_index2 in range(position_index1):
              row2, col2 = self.row_col_coords[position_index2]
              stack2 = input[position_index2]
              for channel_index in self.use_channels:
                  stack1_channel = stack1[:, :, :, channel_index]
                  stack2_channel = stack2[:, :, :, channel_index]
                  z_shift_2 = self.z_translation_weights[position_index2]
                  if not ((row1 == row2 + 1 and col1 == col2) or (row1 == row2 and col1 == col2 + 1)):
                      continue  # non adjacent tiles
                  interped_stack_1 = self._interpolate_stack(z_shift_1 - min_z_trans, stack1_channel)
                  interped_stack_2 = self._interpolate_stack(z_shift_2 - min_z_trans, stack2_channel)
                  if row1 == row2 + 1 and col1 == col2:
                      # stack1 is below stack2
                      overlap1 = interped_stack_1[:, :self.overlap_shape[0], :]
                      overlap2 = interped_stack_2[:, -self.overlap_shape[0]:, :]
                  elif row1 == row2 and col1 == col2 + 1:
                      overlap1 = interped_stack_1[:, :, :self.overlap_shape[1]]
                      overlap2 = interped_stack_2[:, :, -self.overlap_shape[1]:]
                  # o1_mean_sub = overlap1 - tf.reduce_mean(overlap1)
                  # o2_mean_sub = overlap2 - tf.reduce_mean(overlap2)
                  # numer = tf.reduce_sum(o1_mean_sub * o2_mean_sub) ** 2
                  # denom = tf.reduce_sum(o1_mean_sub ** 2) * tf.reduce_sum(o2_mean_sub ** 2)
                  numer = tf.reduce_mean(overlap1 * overlap2) ** 2
                  denom = tf.reduce_mean(overlap1 ** 2) * tf.reduce_mean(overlap2 ** 2)
                  loss += numer / denom
      return -loss

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

def convert_params(intra_stack_params_tensor, stitching_params_tensor, nonempty_pixels):
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
    # compile all params in correct format for stitching--Calculated stitch params should be negated relative
    # to the ones optimized
    stitch_params = np.concatenate([stitching_params_tensor[-1].numpy()[:, None],
                                    np.concatenate([p.numpy() for p in stitching_params_tensor[:-1]], axis=0)], axis=1)
    stitch_params = np.concatenate([-stitch_params[:, 0, None], stitch_params[:, 1:]], axis=1)
    return full_intra_params, stitch_params


def optimize_timepoint(raw_stacks, nonempty_pixels, row_col_coords, overlap_shape, intra_stack_channels,
                       inter_stack_channels, learning_rate=1e-1, stitch_regularization=1, stack_regularization=1, name='image'):
    zyxc_stacks = [np.stack(stack.values(), axis=3) for stack in raw_stacks.values()]
    stacks_layer = IndividualStacksLayer(zyxc_stacks, nonempty_pixels)
    stitch_layer = ImageStitchingLayer(row_col_coords, zyxc_stacks[0].shape, overlap_shape, inter_stack_channels)
    full_model = tf.keras.Sequential([stacks_layer, stitch_layer])
    #model for calculating loss over individual stacks
    individual_stacks_model = tf.keras.Sequential([full_model.get_layer(index=0)])

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    min_loss = np.finfo(np.float).max
    min_loss_iteration = 0
    loss_history = []

    stack_loss_rescale = None
    stitch_loss_rescale = None
    path = '/media/hugespace/henry/lymphosight/optimization_tuning_regularization/'
    with open(path + name + '.txt', 'w') as file:
        for iteration in range(400):
            with tf.GradientTape() as stack_tape:
                stacks = individual_stacks_model(None)
                all_params = full_model.trainable_variables
                intra_stack_params_tensor = [all_params[2 * i] for i in range(len(stacks))]
                stitching_params_tensor = [all_params[2 * i + 1] for i in range(len(stacks))] + [all_params[-1]]

                #compute intra_stack_alignment cost
                stack_loss = tf.zeros((), tf.float32)
                for shifted_stack in stacks:
                    for channel in intra_stack_channels:
                        stack_loss += tf.reduce_mean((
                                            shifted_stack[1:, :, :, channel] - shifted_stack[:-1, :, :, channel]) ** 2)
                if stack_loss_rescale is None: #rescale loss to magnitude one
                    stack_loss_rescale = np.abs(stack_loss.numpy())
                stack_loss = stack_loss / stack_loss_rescale
                #add regularization
                stack_params_vec = tf.concat(intra_stack_params_tensor, axis=0)
                stack_penalty = tf.reduce_mean(tf.abs(stack_params_vec))
                stack_loss = stack_loss + stack_regularization * stack_penalty
            with tf.GradientTape() as full_tape:
                stitching_loss = full_model(None)
                if stitch_loss_rescale is None:
                    stitch_loss_rescale = np.abs(stitching_loss.numpy())
                stitching_loss = stitching_loss / stitch_loss_rescale
                #add regualrization
                stitch_params_single_tensor = tf.concat(
                    [tf.concat((stitching_params_tensor[i], tf.reshape(stitching_params_tensor[-1][i], (1, 1))), axis=1)
                     for i in range(len(stitching_params_tensor) - 1)], axis=0)
                stitch_penalty = tf.reduce_mean(tf.abs(stitch_params_single_tensor))
                stitch_loss = stitch_loss + stitch_regularization * stitch_penalty


            # write out intermeiate images during optimization
            if iteration % 20 == 0:
                full_intra_params, stitch_params = convert_params(intra_stack_params_tensor, stitching_params_tensor, nonempty_pixels)
                # print(stitch_params)
                export_stitched_tiff(raw_stacks, row_col_coords, overlap_shape, full_intra_params, stitch_params,
                                     name='{} iteration{}'.format(name, iteration), path=path)

            stack_grads = stack_tape.gradient(stack_loss, intra_stack_params_tensor)
            # stitch_stack_grads, stitch_grads = full_tape.gradient(stitching_loss, [intra_stack_params_tensor, stitching_params_tensor])
            stitch_grads = full_tape.gradient(stitching_loss, stitching_params_tensor)


            # if iteration == 0:
            #     #rescale gradients to account for different magnitudes corresponding to differences in image intensity
            #     stack_grads_rescale = 0.25 / np.sqrt(np.mean(np.concatenate([np.ravel(g.numpy()) for g in stack_grads]) ** 2))
            #     # rms_stitch_stack_grads = np.sqrt(np.mean(np.concatenate([np.ravel(g.numpy()) for g in stitch_stack_grads]) ** 2))
            #     stitch_grads_rescale = 0.25 / np.sqrt(np.mean(np.concatenate([np.ravel(g.numpy()) for g in stitch_grads]) ** 2))


            # stack_grads = [stack_grads_rescale * g for g in stack_grads]
            # # stitch_stack_grads = [grad_rescale_factor[1] * g for g in stitch_stack_grads]
            # stitch_grads = [stitch_grads_rescale * g for g in stitch_grads]

            #compute rms pixel shifts to monitor progress
            intra_stack_rms_shift = np.sqrt(np.mean(np.concatenate([np.ravel(g.numpy()) for g in intra_stack_params_tensor]) ** 2))
            stitch_rms_shift = np.sqrt(np.mean(np.concatenate([np.ravel(g.numpy()) for g in stitching_params_tensor]) ** 2))

            out = '{},{},{},{}'.format(stitching_loss.numpy(), stack_loss.numpy(), stitch_rms_shift, intra_stack_rms_shift)
            print(out)
            file.write(out + '\n')

            optimizer.apply_gradients(zip(stack_grads, intra_stack_params_tensor), global_step=tf.train.get_or_create_global_step())
            # optimizer.apply_gradients(zip(stitch_stack_grads, intra_stack_params_tensor), global_step=tf.train.get_or_create_global_step())
            optimizer.apply_gradients(zip(stitch_grads, stitching_params_tensor), global_step=tf.train.get_or_create_global_step())

            #make the mean xy shift for stitching 0
            mean_shift = tf.reduce_mean(tf.concat(stitching_params_tensor[:-1], axis=0), axis=0)
            for xy_shift in stitching_params_tensor[:-1]:
                xy_shift.assign(xy_shift - mean_shift)

