# first line: 464
@memory.cache
def resmaple_dist_map(name, target_shape):
    hev_cortex_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/with_surface_masks/distance_transforms.h5',
                                mode='r')
    dist = hev_cortex_file[name]
    return ndi.zoom(dist, np.array(target_shape) / np.array(dist.shape))
