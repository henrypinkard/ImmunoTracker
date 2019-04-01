from pymaricumpiler import convert

magellan_dir = '/Users/henrypinkard/Desktop/Lymphosight/2018-6-2 4 hours post LPS/subregion timelapse_1'
convert(magellan_dir, corrections='fast_register', save_memory=True,
            output_dir=None, output_basename=None, inter_stack_registration_channels=[0], inter_stack_max_z=15)