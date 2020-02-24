import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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


crop1_constant = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/whole LN Constant excitation_1crop1.ims')
crop1_nn = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/whole LN NN excitation_1crop1.ims')
crop2_constant = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/whole LN Constant excitation_1crop2.ims')
crop2_nn = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/whole LN NN excitation_1crop2.ims')
crop3_constant = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/whole LN Constant excitation_1crop3.ims')
crop3_nn = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/whole LN NN excitation_1crop3.ims')

def get_data(dataset, channel):
    return dataset['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel {}'.format(channel)]['Data'][:134]

def make_plot(crop_nn, crop_constant, name):
    all_pix_nn = np.concatenate([np.reshape(get_data(crop_nn, c), [-1, 256*256]) for c in range(6)], axis=1)
    all_pix_const = np.concatenate([np.reshape(get_data(crop_constant, c), [-1, 256*256]) for c in range(6)], axis=1)

    sorted_nn = np.sort(all_pix_nn, axis=1)
    sorted_const = np.sort(all_pix_const, axis=1)

    percent = 0.05
    pixel_size_z = 3.0

    nn_mean_topavg = np.mean(sorted_nn[:, -int(sorted_nn.shape[1] * percent):], axis=1)
    const_mean_topavg = np.mean(sorted_const[:, -int(sorted_nn.shape[1] * percent):], axis=1)

    position_nn = np.flip(np.arange(nn_mean_topavg.shape[0]) * pixel_size_z)
    position_const = np.flip(np.arange(const_mean_topavg.shape[0]) * pixel_size_z)

    position_nn[nn_mean_topavg == 0] = np.nan
    position_const[const_mean_topavg == 0] = np.nan

    nn_mean_topavg[nn_mean_topavg == 0] = np.nan
    const_mean_topavg[const_mean_topavg == 0] = np.nan

    plt.figure()
    plt.plot(const_mean_topavg, position_const)
    plt.plot(nn_mean_topavg, position_nn)
    plt.xlabel('Mean intensity of top 5% of pixels')
    plt.ylabel('Z position')
    plt.legend(['Standard candle calibrated excitation', 'Constant excitation'])
    plt.ylim(0, 400)
    plt.savefig('figures/{}.pdf'.format(name))


make_plot(crop1_nn, crop1_constant, 'crop_1_intensity')
# make_plot(crop2_nn, crop2_constant, 'crop_2_intensity')
# make_plot(crop3_nn, crop3_constant, 'crop_3_intensity')
plt.show()
pass