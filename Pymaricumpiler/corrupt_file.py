from pygellan import MagellanDataset

magellan = MagellanDataset('/media/hugespace/henry/data/lymphosight/raw_data/2018-5-11 BM chimera and XCR1 cherry/subregion OT1 OT2 fast time lapse_1')
z_slices = magellan.get_z_slices_at(2, 12)

for z_slice in z_slices:
	for channel_index in range(6):
		print(z_slice, channel_index)
		magellan.read_image(channel_index=channel_index, z_index=z_slice, t_index=12, pos_index=2)

