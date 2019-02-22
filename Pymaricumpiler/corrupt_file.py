from pygellan import MagellanDataset

magellan = MagellanDataset('/Users/henrypinkard/Desktop/subregion OT1 OT2 fast time lapse_1')

magellan.read_image(channel_index=3, z_index=91, t_index=12, pos_index=2)

#
# for time_index in range(magellan.get_num_frames()):
#     for position_index in range(magellan.get_num_xy_positions()):
# position_index = 2
# time_index = 12
# z_slices = magellan.get_z_slices_at(position_index, time_index=time_index)
# for z_slice in z_slices:
#     for channel_index in range(6):
#         # print(z_slice, channel_index)
#         try:
#             magellan.read_image(channel_index=channel_index, z_index=z_slice, t_index=time_index, pos_index=position_index)
#         except:
#             print('corrupt {} {} {} {}'.format(time_index, channel_index, position_index, z_slice))