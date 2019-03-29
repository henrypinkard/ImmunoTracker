from pymaricumpiler import open_magellan, read_raw_data
from optimization_stitcher import optimize_timepoint
from multiprocessing import Process


def run_optimization(magellan_dir, frame_index, learning_rate, prefilter):
    name = 'frame_{}__lr_{}__prefilter_{}'.format(frame_index, learning_rate, prefilter)

    magellan, metadata = open_magellan(magellan_dir)

    raw_stacks, nonempty_pixels, timestamp = read_raw_data(magellan, metadata, time_index=frame_index,
                            reverse_rank_filter=True, input_filter_sigma=2)
    optimize_timepoint(raw_stacks, nonempty_pixels, metadata['row_col_coords'], metadata['tile_overlaps'],
                       intra_stack_channels=[1, 2, 3, 4, 5], inter_stack_channels=[0, 5],
                       prefilter=prefilter, learning_rate=learning_rate, name=name)

magellan_dir = '/media/hugespace/henry/lymphosight/raw_data/2018-6-2 4 hours post LPS/subregion timelapse_1'

lrs = [1e-2, 1e-1, 5e-1, 1, 5]
prefilter = [True, False]
frame_index = [0, 7]
for learning_rate in lrs:
    for pf in prefilter:
        for t in frame_index:
            print('s')
            p = Process(target=run_optimization, args=(magellan_dir, magellan_dir, t, learning_rate, pf))
            p.start()