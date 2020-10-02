import numpy as np
import h5py
import scipy.ndimage as nd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from util import *
import scipy.ndimage as ndi


# path = '/Users/henrypinkard/Desktop/imaris_analysis/with_surface_masks/'
# files = ['21_cortex.ims', '25_cortex.ims', '25_hevs.ims', '48_cortex.ims', '48_hevs.ims']


path = '/Users/henrypinkard/Desktop/imaris_analysis/with_surface_masks/'
files = ['4445_cropped_to_44.ims']

def distance_map(file):
    print(file)
    dataset = h5py.File(path + file)
    mask = dataset['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel {}'.format(6)]['Data']
    mask = np.copy(mask)
    distances = nd.morphology.distance_transform_edt(np.logical_not(mask), sampling=(4, 0.555, 0.555))
    return distances


distance_maps = [distance_map(f) for f in files]
dist_dict = {files[i][:-4]: distance_maps[i] for i in range(len(files))}

with h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/with_surface_masks/distance_transforms3.h5', 'a') as out_file:
    for name in dist_dict:
        out_file.create_dataset(name=name, data=dist_dict[name])
