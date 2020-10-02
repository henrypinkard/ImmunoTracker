import h5py
import numpy as np
from scipy.spatial import Delaunay
import napari
from tensorflow import keras

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lami_analysis_functions import *



# f = h5py.File(
#     '/Users/henrypinkard/Desktop/Lymphosight/2017 data/2017-4-4 medLN/C_40_600_70_1--Positions as time_GFPCandidates.mat',
#     mode='r')
# interp_points = np.array(f['surfInterpPoints']).T

interp_points = np.load('38_interp_points.npy')
interp_points[:, 2] *= -1 #oritent correctly
interp_points[:, 2] -= np.min(interp_points[:, 2])

tris = Delaunay(interp_points[:, :2])


model = keras.models.load_model('../GFP_LAMI_model', custom_objects={'excitation_power_loss': None})
# model = keras.models.load_model('../e670_LAMI_model', custom_objects = {'excitation_power_loss': None})




#######  Make XZ profile  #######
pixel_size = 20

# y_vals = np.arange(np.min(interp_points, axis=0)[1] + 180, np.max(interp_points, axis=0)[1] - 0, step=pixel_size)
y_vals = np.arange(np.min(interp_points, axis=0)[1], np.max(interp_points, axis=0)[1], step=pixel_size)
z_vals = np.arange(np.min(interp_points, axis=0)[2], np.max(interp_points, axis=0)[2], step=pixel_size)
y_grid, z_grid = np.meshgrid(y_vals, z_vals)
x_val = ((np.max(interp_points, axis=0) - np.min(interp_points, axis=0)) / 2 + np.min(interp_points, axis=0))[0]
x_grid = x_val * np.ones_like(y_grid)

query_points = np.stack([x_grid, y_grid, z_grid], axis=-1)

surf_top = np.apply_along_axis(lambda xy : get_interp_val(xy, tris, interp_points), axis=0,
                           arr=np.stack([x_grid, y_grid], axis=0))

z_diff = surf_top - z_vals[:, None]

#construct feature vectors
feature_vecs = []
for i, point in enumerate(np.reshape(query_points, [-1, 3])):
    print('{} of {}\r'.format(i, query_points.size // 3), end='')
    feature_vecs.append(construct_vector(point, tris, interp_points, fov_position=[0.5, 0.5], brightness=-0.5))


#do predicitions
power_predictions = predict_power(model, np.concatenate(feature_vecs, axis=0))
prediction_img_power = np.reshape(power_predictions, x_grid.shape)



#map surface height
xy_vals = np.stack([np.ones_like(y_vals) * x_val, y_vals], axis=1)
interp_vals = [get_interp_val(xy, tris, interp_points) for xy in xy_vals]
yz_interp_vals = np.stack([ y_vals, interp_vals], axis=1)


dist_to_interp = np.sqrt(np.sum((np.stack([y_grid, z_grid],
                                        axis=2)[..., None] - yz_interp_vals.T) ** 2, axis=2))
min_distance_to_interp = np.min(dist_to_interp, axis=2)

# mask = np.logical_and(z_diff > 0, np.logical_or(z_diff < 300, min_distance_to_interp < 200))

with napari.gui_qt():
    v = napari.Viewer()
    # v.add_image(min_distance_to_interp)
    # v.add_image(mask)
    v.add_image( (z_diff > 0) * prediction_img_power, colormap='inferno')
    # v.add_image(prediction_img_power, colormap='inferno')


# show surf 3d
# with napari.gui_qt():
#     v = napari.Viewer()
#     v.add_surface((interp_points, tris.simplices, -interp_points[:, 2] / 400), colormap='turbo')
#
#


