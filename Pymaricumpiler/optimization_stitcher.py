import tensorflow as tf
import numpy as np
from transformer import spatial_transformer_network
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
      min_z_index = tf.cast(tf.floor(tf.reduce_min(self.z_translation_weights)), tf.int32)
      max_z_index = tf.cast(tf.ceil(tf.reduce_max(self.z_translation_weights)), tf.int32) + self.stack_shape[0]
      stack_z_shape = max_z_index - min_z_index
      # generate interpolated stack based on float z shift
      stack1_z0 = tf.cast(tf.floor(z_shift), tf.int32)
      stack1_z1 = stack1_z0 + 1
      top_stack = tf.concat([tf.zeros((stack1_z0, self.stack_shape[1], self.stack_shape[2])), stack,
                             tf.zeros((stack_z_shape - stack1_z0 - stack_z_shape + 1, self.stack_shape[1], self.stack_shape[2]))], axis=0)
      bottom_stack = tf.concat([tf.zeros((stack1_z1, self.stack_shape[1], self.stack_shape[2])), stack,
                                tf.zeros((stack_z_shape - stack1_z1  - stack_z_shape + 1, self.stack_shape[1],self.stack_shape[2]))], axis=0)
      bottom_weight = z_shift - tf.cast(stack1_z0, tf.float32)
      top_weight = 1 - bottom_weight
      interpolated_stack = top_stack * top_weight + bottom_stack * bottom_weight
      return interpolated_stack

  def call(self, input):
      """
      :param input: nested a list of images indexed by position index then by channel
      :return:
      """
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
                  interped_stack_1 = self._interpolate_stack(z_shift_1, stack1_channel)
                  interped_stack_2 = self._interpolate_stack(z_shift_2, stack2_channel)
                  if row1 == row2 + 1 and col1 == col2:
                      # stack1 is below stack2
                      overlap1 = interped_stack_1[:, :self.overlap_shape[0], :]
                      overlap2 = interped_stack_2[:, -self.overlap_shape[0]:, :]
                  elif row1 == row2 and col1 == col2 + 1:
                      overlap1 = interped_stack_1[:, :, :self.overlap_shape[1]]
                      overlap2 = interped_stack_2[:, :, -self.overlap_shape[1]:]
                  o1_mean_sub = overlap1 - tf.reduce_mean(overlap1)
                  o2_mean_sub = overlap2 - tf.reduce_mean(overlap2)
                  numer = tf.reduce_sum(o1_mean_sub * o2_mean_sub) ** 2
                  denom = tf.reduce_sum(o1_mean_sub ** 2) * tf.reduce_sum(o2_mean_sub ** 2)
                  loss += numer + denom
      return loss


def optimize_timepoint(raw_stacks, nonempty_pixels, row_col_coords, overlap_shape, intra_stack_channels, inter_stack_channels):

    zyxc_stacks = [np.stack(stack.values(), axis=3) for stack in raw_stacks.values()]


    stacks_layer = IndividualStacksLayer(zyxc_stacks, nonempty_pixels)
    stitch_layer = ImageStitchingLayer(row_col_coords, zyxc_stacks[0].shape, overlap_shape, inter_stack_channels)
    full_model = tf.keras.Sequential([stacks_layer, stitch_layer])
    #model for calculating loss over individual stacks
    individual_stacks_model = tf.keras.Sequential([full_model.get_layer(index=0)])


    optimizer = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9)

    min_loss = np.finfo(np.float).max
    min_loss_iteration = 0
    loss_history = []
    for iteration in range(500):
        with tf.GradientTape() as full_tape:
            with tf.GradientTape() as stack_tape:
                stacks = individual_stacks_model(None)
                stack_loss = tf.zeros((), tf.float32)
                for shifted_stack in stacks:
                    stack_loss += tf.reduce_mean((shifted_stack[1:, ...] - shifted_stack[:-1, ...]) ** 2)

            stitching_loss = full_model(None)
            all_params = full_model.trainable_variables
            stack_params = [all_params[2*i] for i in range(len(stacks))]

        print('computing grads')
        stack_grads = stack_tape.gradient(stack_loss, stack_params)
        stitch_grads = full_tape.gradient(stitching_loss, all_params)
        print(stack_grads[0])
        print(stitch_grads[0])

            # data_loss = tf.reduce_mean((shifted[1:, ...] - shifted[:-1, ...]) ** 2)
            # sum_squared_translations = tf.reduce_sum(translation_params**2)
            # #only use sqrt where it doesn't make gradient explode
            # safe_x = tf.where(sum_squared_translations > 1e-2, sum_squared_translations, 1e-2)
            # pixel_shift_2_norm = tf.where(sum_squared_translations > 1e-2, tf.sqrt(safe_x), 0)
            # weight_penalty = pixel_shift_2_norm / translation_params.shape[0].value

            # weight_penalty = tf.reduce_mean(translation_params ** 2)

            # loss = data_loss + weight_penalty * regularization


    # optimizer.apply_gradients(zip(grads, params), global_step=tf.train.get_or_create_global_step())
