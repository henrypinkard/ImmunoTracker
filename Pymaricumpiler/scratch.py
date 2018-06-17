def read_data_cube(channel, frame, position):
    slices = magellan_data.get_slice_indices(channel, frame, position)
    if len(slices) > 0:
        images = []
        for z in slices:
            images.append(magellan_data.read_tile(channel, row, col, z, frame))
    return np.concatenate(images, axis=2)
