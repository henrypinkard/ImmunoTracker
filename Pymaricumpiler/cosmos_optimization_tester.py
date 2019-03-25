from pymaricumpiler import convert

magellan_dir = '/media/hugespace/henry/lymphosight/raw_data/2018-6-2 4 hours post LPS/subregion timelapse_1'
convert(magellan_dir, inter_stack_registration_channels=[5],
                    timepoint_registration_channel=5, n_cores=8, intra_stack_registration_channels=[1, 2, 3, 4, 5],
                    reverse_rank_filter=True, input_filter_sigma=2)


