from pymaricumpiler import open_magellan, read_raw_data
from optimization_stitcher import optimize_timepoint
from multiprocessing import Process, Pool


magellan_dir = '/media/hugespace/henry/lymphosight/raw_data/2018-6-2 4 hours post LPS/subregion timelapse_1'

def run_optimization():
    name = 'different_learning_rates'

    magellan, metadata = open_magellan(magellan_dir)

    raw_stacks, nonempty_pixels, timestamp = read_raw_data(magellan, metadata, time_index=0,
                            reverse_rank_filter=True, input_filter_sigma=2)
    optimize_timepoint(raw_stacks, nonempty_pixels, metadata['row_col_coords'], metadata['tile_overlaps'],
                       intra_stack_channels=[1, 2, 3, 4, 5], inter_stack_channels=[0, 5],
                     name=name)

run_optimization()

# lrs = [1, 3, 9]
# stack_regs = [1e-2, 1]
# stitch_regs = [1e-2, 1e-1, 1]
# params = []
# for learning_rate in lrs:
#     for stack_reg in stack_regs:
#         for stitch_reg in stitch_regs:
#             params.append((learning_rate, stitch_reg, stack_reg))
#
# with Pool(2) as pool:
#     pool.map(run_optimization, params)
    # p = Process(target=run_optimization, args=(magellan_dir, t, learning_rate, pf))
    # p.start()
    # run_optimization(magellan_dir, t, learning_rate, pf)