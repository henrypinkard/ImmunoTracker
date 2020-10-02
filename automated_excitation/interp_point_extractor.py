"""
Open imaris files, visulaize in napari, extract interp points
"""

import napari
import h5py
import pandas
import numpy as np

file = h5py.File('/Users/henrypinkard/Desktop/Lymphosight/LNs with different shapes/38.ims', mode='r')
pixels = file['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data']

with napari.gui_qt():
    v = napari.Viewer()
    v.add_image(pixels)


v.layers[1].save('points_tmp.csv')
df = pandas.read_csv('points_tmp.csv')
interp_points = df[['axis-2', 'axis-1', 'axis-0']].to_numpy()
interp_points *= np.array([0.55, 0.55, 4])

np.save('validation_ln_interp_points', interp_points)