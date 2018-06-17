from magellan_data import MagellanJavaWrapper
from imariswriter import ImarisJavaWrapper
import numpy as np
import os

data_dir = '/Users/henrypinkard/Desktop/LNData/subregion timelapse_2'

#open magellan dataset
magellan_dataset = MagellanJavaWrapper(data_dir)
#read some useful metadata
byte_depth = magellan_dataset.byte_depth
num_frames = magellan_dataset.num_frames
num_positions = magellan_dataset.num_positions
min_z_index = magellan_dataset.slice_limits[0]
max_z_index = magellan_dataset.slice_limits[1]
channel_names = magellan_dataset.channel_names
coords_by_pos_indices = magellan_dataset.grid_coords_from_position_index
overlap_x = magellan_dataset.overlap_x
overlap_y = magellan_dataset.overlap_y
tile_width = magellan_dataset.tile_width
tile_height = magellan_dataset.tile_height
num_rows = magellan_dataset.num_rows
num_cols = magellan_dataset.num_cols
pixel_size_xy_um = magellan_dataset.pixel_size_xy_um
pixel_size_z_um = magellan_dataset.pixel_size_z_um


#compute directory and name for output imaris file
dir = os.sep.join(data_dir.split(os.sep)[:-1]) #parent directory of magellan
name = data_dir.split(os.sep)[-1] #same name as magellan acquisition
#compute dimesnions of stitched image
imaris_size_x = num_cols * (tile_width - overlap_x)
imaris_size_y = num_rows * (tile_height - overlap_y)
imaris_size_z = max_z_index - min_z_index + 1
num_channels = len(channel_names)

with ImarisJavaWrapper(dir, name, (imaris_size_x, imaris_size_y, imaris_size_z), byte_depth, num_channels,
                       num_frames, pixel_size_xy_um, pixel_size_z_um) as writer:
    for time_index in range(num_frames):
        for channel_index in range(num_channels):
            for z_index in range(min_z_index, max_z_index + 1):
                #stitch all tiles together
                stitched_image = np.zeros(shape=(imaris_size_y, imaris_size_x), dtype=np.uint16 if byte_depth == 2 else np.uint8)
                #add in whatever tiles are present, leaving zeros elsewhere
                #TODO: make background pixel values?
                elapsed_time_ms = ''
                for position_index in range(num_positions):
                    row, col = coords_by_pos_indices[position_index]
                    image_and_metadata = magellan_dataset.read_tile(channel_index=channel_index, position_index=position_index,
                                                                    z_index=z_index, time_index=time_index, return_metadata=True)
                    if image_and_metadata is None:
                        continue #no tile collected at this position
                    image, metadata = image_and_metadata
                    central_square = image[overlap_y//2:-overlap_y//2, overlap_x//2:-overlap_x//2]
                    stitched_image[row * (tile_height - overlap_y):(row + 1) * (tile_height - overlap_y), col * (tile_width - overlap_x):(col + 1) * (tile_width - overlap_x)] = central_square
                    if elapsed_time_ms == '':
                        elapsed_time_ms = metadata['ElapsedTime-ms']

                #add image to imaris writer
                print('Frame: {} of {}, Channel: {} of {}, Slice: {} of {}'.format(time_index+1, num_frames, channel_index+1,
                                                                                   num_channels, z_index - min_z_index+1, imaris_size_z))
                writer.write_z_slice(stitched_image, z_index - min_z_index, channel_index, time_index, elapsed_time_ms)

print('Finshed!')