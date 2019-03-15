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
from pymaricumpiler import open_magellan, read_raw_data, estimate_background
import time


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
    self.abs_translations = self.add_variable("abs_translations", [self.num_z_slices, 2, 1], dtype=tf.float32,
                                              initializer=tf.keras.initializers.zeros)

  def call(self, input):
      identities = tf.eye(2, 2, [self.abs_translations.shape[0].value])
      #normalizing so translations can be kept in pixel coordinates
      normalized_translations = self.abs_translations / np.array([self.height / 2, self.width / 2])[None, :, None]
      affines = tf.concat([identities, normalized_translations], axis=2)
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

    start_time = time.start()

    #TODO:
    position_index = 1

    all_channel_stack = np.stack([raw_stacks[position_index][channel] for channel in use_channels], axis=3)
    all_channel_stack_valid = all_channel_stack[nonempty_pixels[position_index]].astype(np.float32) #use only slices where data was collected
    filtered = np.apply_over_axes(lambda img: filters.gaussian_filter(img, sigma), all_channel_stack_valid, [0, 3])
    #normalize by total intensity in all channels so dimmer parts of stack don't have weaker gradients
    normalizations = np.mean(np.reshape(filtered, (filtered.shape[0], -1)), axis=1) - np.mean(backgrounds)
    if normalize:
        normalized_stack = filtered / normalizations[:, None, None, None]
    else:
        normalized_stack = filtered
    model = tf.keras.Sequential([ImageTranslationLayer(normalized_stack, max_shift)])

    #Optimize
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    min_loss = np.finfo(np.float).max
    min_loss_iteration = 0
    loss_history = []
    for iteration in range(3000):
        with tf.GradientTape(persistent=True) as tape:
            shifted = model(None)
            translation_params = model.trainable_variables[0]
            data_loss = tf.reduce_sum(tf.abs(shifted[1:, ...] - shifted[:-1, ...]))
            weight_penalty = tf.reduce_mean(tf.abs(translation_params))
            loss = data_loss
                    # + 1e0*weight_penalty
        loss_numpy = loss.numpy()
        loss_history.append(loss_numpy)
        if loss_numpy < min_loss:
            min_loss = loss_numpy
            min_loss_iteration = iteration
        if iteration > min_loss_iteration + 20:
            break
        print('iteration {}, loss {}'.format(iteration, loss.numpy()))
        #get gradient and take step
        grads = tape.gradient(loss, [translation_params])
        optimizer.apply_gradients(zip(grads, [translation_params]), global_step=tf.train.get_or_create_global_step())
    corrections = np.flip(translation_params.numpy()[..., 0], axis=1)


    #TODO: add regularization to bias solutions towards 0 when little data?

    name = 'mom_{}__lr_{}__norm_{}__sigma_{}'.format(momentum,learning_rate,normalize, sigma)
    path = '/media/hugespace/henry/data/lymphosight/optmization_tuning/'
    with open(name + '.txt', 'w') as file:
        file.write(str(loss_history))
        elapsed = time.time() - start_time
        file.write('elapsed: {}'.format(elapsed))

    fixed_stacked = np.stack([apply_intra_stack_registration(raw_stacks[position_index][channel][nonempty_pixels[position_index]], corrections) for channel in range(6)], axis=0)
    exporttiffstack(np.reshape((fixed_stacked), (fixed_stacked.shape[0]*fixed_stacked.shape[1],
                    fixed_stacked.shape[2],fixed_stacked.shape[3])).astype(np.uint8), name=name)




magellan_dir = '/Users/henrypinkard/Desktop/Lymphosight/2018-6-2 4 hours post LPS/subregion timelapse_1'
reverse_rank_filter=True
input_filter_sigma=2
intra_stack_registration_channels=[1, 2, 3, 4, 5]

magellan, metadata = open_magellan(magellan_dir)
#iterate through all time points to compute all needed stitching and registration params
all_params = []
previous_stitched = None
backgrounds=None
stitched_image_size=None

raw_stacks, nonempty_pixels, timestamp = read_raw_data(magellan, metadata, time_index=7,
                            reverse_rank_filter=reverse_rank_filter, input_filter_sigma=input_filter_sigma)

backgrounds = estimate_background(raw_stacks, nonempty_pixels)

lrs = [1e-6, 2e-5, 6e-5, 2e-4]
sigmas = [0, 3, 7, 10]
momentums = [0.9, 0.95, 0.99, 0.999]
normalizes = [True, False]
for learning_rate in lrs:
    for momentum in momentums:
        for normalize in normalizes:
            for sigma in sigmas:
                registration_params = optimize_intra_stack_registrations(raw_stacks, nonempty_pixels,
                        np.max(metadata['tile_overlaps']),  backgrounds=backgrounds,
                        use_channels=intra_stack_registration_channels)
