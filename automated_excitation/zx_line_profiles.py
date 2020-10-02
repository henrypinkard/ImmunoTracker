import h5py
import numpy as np
from scipy.spatial import Delaunay
import napari
from tensorflow import keras
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# settings for exporting plots to illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# make text on figures look good
SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 28
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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



######## z profile across central point #########
# max_depth = 250
# n_points = 15
# central_profile = make_central_profile(model, interp_points, tris,
#                                        max_depth=max_depth, n_points=n_points)
#
# #plot and fit center profile
# # z_position = np.linspace(0, max_depth, n_points)
# # plt.figure()
# # plt.plot(z_position, central_profile)
# # plt.show()
#
# # Do exponential fit
# z_position = np.linspace(0, max_depth, n_points)
# log_y_data = np.log(central_profile - np.min(central_profile) + 0.01)
#
# exp_fit = np.polyfit(z_position[:9], log_y_data[:9], 1)
# # lin_fit = np.polyfit(z_position[2:10])
# plt.figure()
# plt.plot(z_position, central_profile, "o")
# plt.plot(z_position, np.min(central_profile) + np.exp(exp_fit[1] + z_position * exp_fit[0]))
# # plt.semilogy(z_position,  np.exp(curve_fit[1] + z_position * curve_fit[0]))
# plt.ylim([0, 1.1])
# plt.legend(['predicted power', 'exponential fit'])
# plt.ylabel('Normalized excitation laser power')
# plt.xlabel('Depth into sample (um)')
# plt.show()
# plt.savefig('figures/exponential_fit.pdf')



#### Shift pattern until translatio offset causes problem

x_val = ((np.max(interp_points, axis=0) - np.min(interp_points, axis=0)) / 2
                                    + np.min(interp_points, axis=0))[0]
#find bounding y values
for y_min in range(-1000000, 100000):
    if get_interp_val([x_val, y_min], tris, interp_points) is not None:
        break
    # y_min+=1 # just to make sure
for y_max in range(y_min, 100000):
    if get_interp_val([x_val, y_max], tris, interp_points) is None:
        y_max -= 1
        break

y_center = (y_max - y_min) / 2 + y_min

y_vals = np.arange(y_center - 200, y_center + 400, 100)
# y_vals = np.arange(y_center - 400, y_center + 600, 20)


xys = np.stack([x_val * np.ones_like(y_vals),  y_vals], axis=1)


z_vals = [get_interp_val(xy, tris, interp_points) for xy in xys]
depth_profiles = [make_depth_profile(model, xy, interp_points, tris, max_depth=250) for xy in xys]
# depth_profiles = [p for p in depth_profiles if p is not None]

plt.figure()
for z_offset, prof in zip(z_vals, depth_profiles):
    # plt.plot(np.linspace(0, 250, prof.size), gaussian_filter1d(np.ravel(prof), 1, mode='nearest'))
    plt.plot(np.linspace(0, 250, prof.size) + z_offset, gaussian_filter1d(np.ravel(prof), 1, mode='nearest'))
plt.legend([str(i * 100) for i in range(len(depth_profiles))])
plt.show()
plt.savefig('figures/depth profile.pdf')


# hroz_profile_100_um = np.stack(depth_profiles)[:, 5, 0]
# horz_profile_200_um = np.stack(depth_profiles)[:, 10, 0]
# plt.figure()
# plt.plot(np.arange(hroz_profile_100_um.size) * 20, gaussian_filter1d(hroz_profile_100_um,2, mode='nearest'))
# plt.plot(np.arange(horz_profile_200_um.size) * 20, gaussian_filter1d(horz_profile_200_um, 2, mode='nearest'))
# plt.legend(['100 um depth', '200 um depth'])
# plt.show()



######## Shift pattern until profile changes shape #######

x_val = ((np.max(interp_points, axis=0) - np.min(interp_points, axis=0)) / 2
                                    + np.min(interp_points, axis=0))[0]
#find bounding y values
for y_min in range(100000):
    if get_interp_val([x_val, y_min], tris, interp_points) is not None:
        break
    # y_min+=1 # just to make sure
for y_max in range(y_min, 100000):
    if get_interp_val([x_val, y_max], tris, interp_points) is None:
        y_max -= 1
        break

y_center = (y_max - y_min) / 2 + y_min

y_vals = np.arange(y_center - 400, y_center + 600, 200)

# y_vals = np.linspace(y_min + 50, y_max - 50, 6)

xys = np.stack([x_val * np.ones_like(y_vals),  y_vals], axis=1)


z_vals = [get_interp_val(xy, tris, interp_points) for xy in xys]
depth_profiles = [make_depth_profile(model, xy, interp_points, tris, max_depth=250) for xy in xys]
# depth_profiles = [p for p in depth_profiles if p is not None]

plt.figure()
for z_offset, prof in zip(z_vals, depth_profiles):
    # plt.plot(np.linspace(0, 250, prof.size), gaussian_filter1d(np.ravel(prof), 1, mode='nearest'))
    plt.plot(np.linspace(0, 250, prof.size), gaussian_filter1d(np.ravel(prof), 1, mode='nearest'))
plt.legend([str(i * 200) for i in range(len(depth_profiles))])
plt.show()
plt.savefig('figures/excitation profile shape.pdf')

