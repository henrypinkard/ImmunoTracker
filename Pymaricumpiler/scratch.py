from pymaricumpiler import convert
import os
import csv
import numpy as np
from pygellan import MagellanDataset

magellan_dir = '/Users/henrypinkard/Desktop/xcr1 32 post LPS_1'

magellan = MagellanDataset(magellan_dir)
#find indices for the first image to read time metadata
channel = 0
position = list(magellan.p_t_z_c_tree.keys())[0]
z = list(magellan.p_t_z_c_tree[position][channel].keys())[0]
frame = 0
image, metadata = magellan.read_image(channel_index=channel, z_index=z,
                                 t_index=frame, pos_index=position, read_metadata=True)
time = metadata['Time'][1 +metadata['Time'].find(' '): -2]
frames = magellan.get_num_frames()
positions = magellan.get_num_xy_positions()
min_z, max_z = magellan.get_min_max_z_index()
z_size = max_z - min_z
print('{},{},{},{}'.format(time, frames, positions, z_size))

# convert(magellan_dir, do_registrations=False, do_translations=False, output_dir=None, output_basename=None,
#         background=10, inter_stack_registration_channel=0, timepoint_registration_channel=0, n_cores=8):