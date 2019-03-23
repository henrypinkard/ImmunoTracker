from pygellan import MagellanDataset
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import filters

def open_magellan(path):
    """
    open a magellan dataset on disk and read all appropriate metadata fields
    :param path: path to top level magellan folder
    :return:
    """
    magellan = MagellanDataset(path)
    metadata = {}
    if magellan.summary_metadata['PixelType'] == 'GRAY8':
        metadata['byte_depth'] = 1
    else:
        metadata['byte_depth'] = 2
    metadata['num_positions'] = magellan.get_num_xy_positions()
    min_z_index, max_z_index = magellan.get_min_max_z_index()
    metadata['min_z_index'] = min_z_index
    metadata['max_z_index'] = max_z_index
    metadata['num_channels'] = len(magellan.summary_metadata['ChNames'])
    metadata['tile_overlaps'] = np.array([magellan.summary_metadata['GridPixelOverlapY'],
                                          magellan.summary_metadata['GridPixelOverlapX']])
    metadata['tile_shape'] = np.array([magellan.summary_metadata['Height'], magellan.summary_metadata['Width']])
    metadata['pixel_size_xy_um'] = magellan.summary_metadata['PixelSize_um']
    metadata['pixel_size_z_um'] = magellan.summary_metadata['z-step_um']
    metadata['num_frames'] = magellan.get_num_frames()
    num_rows, num_cols = magellan.get_num_rows_and_cols()
    metadata['num_rows'] = num_rows
    metadata['num_cols'] = num_cols
    metadata['row_col_coords'] = np.array(magellan.row_col_tuples)
    return magellan, metadata

def read_raw_data(magellan, metadata, time_index, reverse_rank_filter=False, input_filter_sigma=None):
    """
    read raw data, store in 3D arrays for each channel at each position
    :param magellan:
    :param metadata:
    :param reverse_rank_filter:
    :return:
    """
    elapsed_time_ms = ''
    raw_stacks = {}
    nonempty_pixels = {}
    for position_index in range(metadata['num_positions']):
        raw_stacks[position_index] = {}
        nonempty_pixels[position_index] = {}
        print('Reading in frame {}, position {}'.format(time_index, position_index))
        for channel_index in range(metadata['num_channels']):
            raw_stacks[position_index][channel_index] = np.zeros((metadata['max_z_index'] -
                    metadata['min_z_index'] + 1, *metadata['tile_shape']),
                                                dtype= np.uint8 if metadata['byte_depth'] == 1 else np.uint16)
            nonempty_pixels[position_index] = (metadata['max_z_index'] - metadata['min_z_index'] + 1)*[False]
            for z_index in range(raw_stacks[position_index][channel_index].shape[0]):
                if not magellan.has_image(channel_index=channel_index, pos_index=position_index,
                                        z_index=z_index + metadata['min_z_index'], t_index=time_index):
                    continue
                image, image_metadata = magellan.read_image(channel_index=channel_index, pos_index=position_index,
                                z_index=z_index + metadata['min_z_index'], t_index=time_index, read_metadata=True)
                if reverse_rank_filter:
                    #do final step of rank fitlering
                    image = ndi.percentile_filter(image, percentile=15, size=3)
                if input_filter_sigma is not None:
                    image = filters.gaussian_filter(image.astype(np.float), input_filter_sigma)

                #add in image
                raw_stacks[position_index][channel_index][z_index] = image
                nonempty_pixels[position_index][z_index] = True
                if elapsed_time_ms == '':
                    elapsed_time_ms = image_metadata['ElapsedTime-ms']
    return raw_stacks, nonempty_pixels, elapsed_time_ms
