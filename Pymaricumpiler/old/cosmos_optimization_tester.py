from pymaricumpiler import convert
from pymaricumpiler import open_magellan, read_raw_data
from optimization_stitcher import optimize_timepoint
from multiprocessing import Process, Pool

magellan_dir = '/media/hugespace/henry/lymphosight/raw_data/2018-6-2 4 hours post LPS/subregion timelapse_1'
magellan, metadata = open_magellan(magellan_dir)

raw_stacks, nonempty_pixels, timestamp = read_raw_data(magellan, metadata, time_index=0,
                                                       reverse_rank_filter=True, input_filter_sigma=2)
optimize_timepoint(raw_stacks, nonempty_pixels, metadata['row_col_coords'], metadata['tile_overlaps'],
                   intra_stack_channels=[1, 2, 3, 4, 5], inter_stack_channels=[0, 5],
                    learning_rate=0.1, name='test')

