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
# interp_points[:, 2] *= -1 #oritent correctly
interp_points[:, 2] -= np.min(interp_points[:, 2])

tris = Delaunay(interp_points[:, :2])


model = keras.models.load_model('../GFP_LAMI_model', custom_objects={'excitation_power_loss': None})
# model = keras.models.load_model('../e670_LAMI_model', custom_objects = {'excitation_power_loss': None})


######## predict single point #########
# point = np.array([ 729.76806715,  550.44873632, -50])
# p1 = predict_power(model, construct_vector(point, tris, interp_points, fov_position=[0.5, 0.5], brightness=-1))


######## z profiles across a grid of XY points #########
max_depth = 250
n_points = 20
central_profile = make_central_profile(interp_points, tris,
                                       max_depth=max_depth, n_points=n_points)

#plot and fit center profile
z_position = np.linspace(0, max_depth, n_points)
plt.figure()
plt.plot(z_position, central_profile)
plt.show()

# Do exponential fit
z_position = np.linspace(0, max_depth, n_points)
log_y_data = np.log(central_profile - np.min(central_profile) + 0.01)

exp_fit = np.polyfit(z_position[:10], log_y_data[:10], 1)
# lin_fit = np.polyfit(z_position[2:10])
plt.figure()
plt.plot(z_position, central_profile, "o")
plt.plot(z_position, np.min(central_profile) + np.exp(exp_fit[1] + z_position * exp_fit[0]))
# plt.semilogy(z_position,  np.exp(curve_fit[1] + z_position * curve_fit[0]))
plt.ylim([0, 1])
plt.show()



##### visualize 2 differtn profiles

center_position = (np.max(interp_points[:, :2], axis=0) - np.min(interp_points[:, :2],
                                    axis=0)) / 2 + np.min(interp_points[:, :2], axis=0)

p_center = make_depth_profile(model, center_position, interp_points, tris)
p_300 = make_depth_profile(model, [center_position[0] + 600, center_position[1]], interp_points, tris)

plt.figure()
plt.plot(p_center)
plt.plot(p_300)
plt.show()


#### compute different profiles along a regular grid
xy_step_size = 100 # Step size for XY

x_vals = np.arange(np.min(interp_points, axis=0)[0], np.max(interp_points, axis=0)[0], step=xy_step_size)
y_vals = np.arange(np.min(interp_points, axis=0)[1], np.max(interp_points, axis=0)[1], step=xy_step_size)

x_grid, y_grid = np.meshgrid(x_vals, y_vals)

is_in_bounds = np.apply_along_axis(lambda xy: get_interp_val(xy, tris, interp_points) is not None,
                                   axis=0, arr=np.stack([x_grid, y_grid]))

#measure profiles over grid of points spanning entire sample
depth_profiles = []
for i, xy in enumerate(np.reshape(np.stack([x_grid, y_grid], axis=-1), [-1, 2])):
    print('z position profile {} of {}\r'.format(i, x_grid.size), end='')
    if not np.ravel(is_in_bounds)[i]:
        continue
    depth_profiles.append(make_depth_profile(model, xy, interp_points, tris, from_surface_top=False))


discrepancy = np.abs(np.concatenate(depth_profiles, axis=1) - central_profile[:, None]).T

plt.hist(np.max(discrepancy, axis=1), 80)
plt.show()
np.sum(np.max(discrepancy, axis=1) < 0.25) / np.max(discrepancy, axis=1).size

plt.figure()
for j, prof in enumerate(depth_profiles):
    plt.plot(prof)
plt.plot(central_profile, 'k')
plt.show()

