import jax.numpy as np

def _generate_grid(image, zyx_translations=None, yx_translations=None):
    """
    Generate Nx3 (# of pixels) x (zyx) gird of coordinates to sample new pixels
    :param image:
    :param zyx_translations: 1 x 3 global translation for the whole stack
    :param yx_translations: (# z) x 2 (y,x ) slice by slice registrations
    :return:
    """
    #get zyx coords of all points
    zyx = np.meshgrid(*[np.arange(d) for d in image.shape], indexing='ij')
    #flatten into nx3 vector
    n_by_zyx = np.stack([np.ravel(d) for d in zyx], axis=1)
    if zyx_translations is not None:
        #apply globabl yxz translations
        n_by_zyx = n_by_zyx + zyx_translations
    if yx_translations is not None:
        #apply per image shifts
        #reshape to make first axis z again
        per_slice_coords = np.reshape(n_by_zyx, [image.shape[0], -1, 3])
        #add in empty z coord
        per_slice_shifts = np.concatenate([np.zeros((yx_translations.shape[0], 1)), yx_translations], axis=1)
        per_slice_coords += np.reshape(per_slice_shifts, [yx_translations.shape[0], -1, 3])
        n_by_zyx = np.reshape(per_slice_coords, [-1, 3])
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
    floor_indices = np.floor(n_by_zyx)
    ceil_indices = np.ceil(n_by_zyx)
    combos = [[floor_indices[:, 0], floor_indices[:, 1], floor_indices[:, 2]],
             [floor_indices[:, 0], floor_indices[:, 1], ceil_indices[:, 2]],
             [floor_indices[:, 0], ceil_indices[:, 1], floor_indices[:, 2]],
             [floor_indices[:, 0], ceil_indices[:, 1], ceil_indices[:, 2]],
             [ceil_indices[:, 0], floor_indices[:, 1], floor_indices[:, 2]],
             [ceil_indices[:, 0], floor_indices[:, 1], ceil_indices[:, 2]],
             [ceil_indices[:, 0], ceil_indices[:, 1], floor_indices[:, 2]],
             [ceil_indices[:, 0], ceil_indices[:, 1], ceil_indices[:, 2]] ]
    corner_indices = np.stack([np.stack(c, axis=1) for c in combos], axis=1)
    #reshape to n x 3
    int_indices_flat = np.reshape(corner_indices, [-1, 3])
    #mkae negatives really big so they get clipped and sample with the default value
    int_indices_flat = int_indices_flat.astype(np.uint32)

    #pad 1 pixel border on top of image with zeros so you can index here for anything out of bounds
    padded = image
    padded = np.concatenate([padded, fill_val*np.ones((1, padded.shape[1], padded.shape[2]))], axis=0)
    padded = np.concatenate([padded, fill_val*np.ones((padded.shape[0], 1, padded.shape[2]))], axis=1)
    padded = np.concatenate([padded, fill_val*np.ones((padded.shape[0], padded.shape[1], 1))], axis=2)
    valid_indces = np.minimum(int_indices_flat, shape_array[None, :].astype(np.uint32))
    pixel_vals_flat = padded[valid_indces[:, 0], valid_indces[:, 1], valid_indces[:, 2]]

    pixel_values = np.reshape(pixel_vals_flat, [-1, 8])

    #calculate weights: N x 8
    ceil_weights = n_by_zyx - floor_indices
    floor_weights = 1 - ceil_weights
    corner_weights = np.stack([floor_weights[:, 0] * floor_weights[:, 1] * floor_weights[:, 2],
                                 floor_weights[:, 0] * floor_weights[:, 1] * ceil_weights[:, 2],
                                 floor_weights[:, 0] * ceil_weights[:, 1] * floor_weights[:, 2],
                                 floor_weights[:, 0] * ceil_weights[:, 1] * ceil_weights[:, 2],
                                 ceil_weights[:, 0] * floor_weights[:, 1] * floor_weights[:, 2],
                                 ceil_weights[:, 0] * floor_weights[:, 1] * ceil_weights[:, 2],
                                 ceil_weights[:, 0] * ceil_weights[:, 1] * floor_weights[:, 2],
                                 ceil_weights[:, 0] * ceil_weights[:, 1] * ceil_weights[:, 2]], axis=1)

    #add together to get pixels values
    interpolated_pixels = np.sum(pixel_values * corner_weights, axis=1)
    return np.reshape(interpolated_pixels, image.shape)


def interpolate_stack(img, fill_val, zyx_translations=None, yx_translations=None):
    """
    Bilinear resample and interpolate an image stack
    :param image: zyx image stack
    :param fill_val: what to make the background where no data available
    :param zyx_translations: 3 element vector for glabl translation of the stack
    :param yx_translations: vector of slice by slice registration (equal to 2 * image.shape[0])
    :return: resampled image stack
    """
    yx_translations = np.reshape(yx_translations, [-1, 2])
    grid = _generate_grid(img, yx_translations=yx_translations, zyx_translations=zyx_translations)
    resampled = _sample_pixels(img, grid, fill_val=fill_val)
    return resampled
