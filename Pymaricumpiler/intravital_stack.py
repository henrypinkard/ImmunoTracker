import tensorflow as tf
tf.enable_eager_execution()

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from transformer import spatial_transformer_network
from PIL import Image
from scipy.ndimage import filters
from scipy import signal


class ImageTranslationLayer(tf.keras.layers.Layer):
  #batch size is the number of slices in the z stack
  def __init__(self, image, max_shift):
      super(ImageTranslationLayer, self).__init__()
      self.image = tf.reshape(tf.convert_to_tensor(image), shape=image.shape)
      # compute averge intensity at each slice to get normalization
      self.num_z_slices = image.shape[0]
      self.height = image.shape[1]
      self.width = image.shape[2]
      self.max_shift = max_shift

  def build(self, input_shape):
    # The build method gets called the first time your layer is used.
    # Creating variables on build() allows you to make their shape depend
    # on the input shape and hence removes the need for the user to specify
    # full shapes. It is possible to create variables during __init__() if
    # you already know their full shapes.
    self.translation_weights = self.add_variable("abs_translations", [self.num_z_slices * 2], dtype=tf.float32,
                                              initializer=tf.keras.initializers.zeros)

  def call(self, input):
      abs_translations = tf.reshape(self.translation_weights, [self.num_z_slices, 2])
      identities = tf.eye(2, 2, [abs_translations.shape[0].value])
      #normalizing so translations can be kept in pixel coordinates
      normalized_translations = abs_translations / np.array([self.height / 2, self.width / 2])[None, :]
      affines = tf.concat([identities, normalized_translations[:, :, None]], axis=2)
      transformed = spatial_transformer_network(self.image, tf.reshape(affines, [-1, 6]))
      crop_offset = tf.convert_to_tensor(self.max_shift // 2)
      self.cropped_and_transated = transformed[:, crop_offset:-crop_offset, crop_offset:-crop_offset, :]
      return self.cropped_and_transated

def exporttiffstack(datacube, name='export', path='/Users/henrypinkard/Desktop/'):
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
    path = "{}{}.tif".format(path,name)
    imlist[0].save(path, compression="tiff_deflate", save_all=True, append_images=imlist[1:])

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
        registered_stack = np.ones_like(single_channel_stack) * background
        for slice in range(registrations.shape[0]):
            # need to negate to make it work right
            reg = -np.round(registrations).astype(np.int)[slice]
            orig_slice = single_channel_stack[slice, ...]
            reg_slice = registered_stack[slice, ...]
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
        return registered_stack

def optimize_intra_stack_registrations(raw_stacks, nonempty_pixels, max_shift, backgrounds, use_channels,
                                       learning_rate=2e-5, sigma=7, momentum=0.99, normalize=True):
    """
     Compute registrations for each z slice within a stack using method based on cross correaltion followed by gaussian
     filtering and MLE fitting
     :param channel_stack:
     :param nonempty_pixels:
     :param max_shift:
     :param background:
     :param use_channels:
     :return:
     """
    registration_params = []
    for position_index in raw_stacks.keys():

        #TODO:
        position_index = 1

        all_channel_stack = np.stack([raw_stacks[position_index][channel] for channel in use_channels], axis=3)
        all_channel_stack_valid = all_channel_stack[nonempty_pixels[position_index]].astype(
            np.float32)  # use only slices where data was collected
        filtered = np.zeros_like(all_channel_stack_valid)
        for slice in range(all_channel_stack_valid.shape[0]):
            for channel in range(all_channel_stack_valid.shape[3]):
                # filtered[slice, :, :, channel] = filters.gaussian_filter(all_channel_stack_valid[slice, :, :, channel],
                #                                                          sigma)
                filtered[slice, :, :, channel] = signal.medfilt2d(all_channel_stack_valid[slice, :, :, channel], 5)

        # normalize by total intensity in all channels so dimmer parts of stack don't have weaker gradients
        normalizations = np.mean(np.reshape(filtered, (filtered.shape[0], -1)), axis=1) - np.mean(backgrounds)
        if normalize:
            normalized_stack = filtered / normalizations[:, None, None, None]
        else:
            normalized_stack = filtered

        model = tf.keras.Sequential([ImageTranslationLayer(normalized_stack, max_shift)])

        def optimize(learning_rate=3e-5, stopping_iterations=20, regularization=1e4):
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            min_loss = np.finfo(np.float).max
            min_loss_iteration = 0
            loss_history = []
            for iteration in range(25):
                with tf.GradientTape() as tape:
                    shifted = model(None)
                    translation_params = model.trainable_variables[0]
                    data_loss = tf.reduce_mean((shifted[1:, ...] - shifted[:-1, ...])**2)
                    squared_pixel_shifts = tf.reduce_sum(tf.reshape(translation_params, [-1, 2]) **2, axis=1)
                    #only use sqrt where it doesn't make gradient explode
                    safe_x = tf.where(squared_pixel_shifts > 1e-2, squared_pixel_shifts, 1e-2*tf.ones_like(squared_pixel_shifts))
                    safe_f = tf.zeros_like
                    pixel_shifts = tf.where(squared_pixel_shifts < 1e-2, tf.sqrt(safe_x), safe_f(squared_pixel_shifts))
                    weight_penalty = tf.reduce_mean(pixel_shifts)
                    loss = data_loss + weight_penalty*regularization
                grads = tape.gradient(loss, [translation_params])

                rms_shift = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.reshape(translation_params, [-1, 2]) ** 2, axis=1))).numpy()
                optimizer.apply_gradients(zip(grads, [translation_params]), global_step=tf.train.get_or_create_global_step())
                #record loss and maybe break loop
                loss_numpy = loss.numpy()
                loss_history.append(loss_numpy)
                if loss_numpy < min_loss:
                    min_loss = loss_numpy
                    min_loss_iteration = iteration
                if iteration > min_loss_iteration + stopping_iterations:
                    break
                print('iteration {}, data loss {} \t\t reg loss {} \t\t rms shift {}'.format(iteration, data_loss.numpy(),
                                          regularization * weight_penalty.numpy(), rms_shift))
            corrections = np.flip(np.reshape(translation_params.numpy(), [-1, 2]), axis=1)
            return corrections

        corrections = optimize(learning_rate=2e2, regularization=1e-2, momentum=0.95)

        learning_rate = 5e-6
        while learning_rate >= 1e-8:
            print('optimizing: learning rate {}'.format(learning_rate))
            corrections = optimize(learning_rate=learning_rate)
            learning_rate =learning_rate * 1e-2

        s = 40

        #TODO: add regularization to bias solutions towards 0 when little data?

        # np.round(translation_params.numpy())         #for viewing integer translations

        # exporttiffstack(shifted.numpy()[...,4].astype(np.uint8), 'optimized_one_channel')
        # exporttiffstack(all_channel_stack[...,4].astype(np.uint8), 'raw_one_channel')
        # exporttiffstack(apply_intra_stack_registration(all_channel_stack[..., 4], corrections).astype(np.uint8), 'hp_correct_one_channel')

        name = 'mom_{}__lr_{}__norm_{}__sigma_{}'.format(momentum,learning_rate,normalize, sigma)
        with open(name + '.txt', 'w') as file:
            file.write(str(loss_history))

        raw_stacked = np.stack([raw_stacks[position_index][channel][nonempty_pixels[position_index]] for channel in range(6)], axis=0)
        fixed_stacked = np.stack([apply_intra_stack_registration(raw_stacks[position_index][channel][nonempty_pixels[position_index]], corrections) for channel in range(6)], axis=0)
        exporttiffstack(np.reshape((fixed_stacked), (fixed_stacked.shape[0]*fixed_stacked.shape[1],
                        fixed_stacked.shape[2],fixed_stacked.shape[3])).astype(np.uint8), name='regged')

        exporttiffstack(np.reshape((raw_stacked), (fixed_stacked.shape[0]*fixed_stacked.shape[1],
                        fixed_stacked.shape[2],fixed_stacked.shape[3])).astype(np.uint8), name='raw2')


        #TODO
        raise Exception('Dont do the rest')



        #0s for empty slices, insert the calculated ones for others
        registrations = np.zeros((raw_stacks[position_index][0].shape[0]), 2)
        registrations[nonempty_pixels[position_index]] = corrections
        registration_params.append(registrations)

        # registrations[nonempty_pixels[position_index]]
