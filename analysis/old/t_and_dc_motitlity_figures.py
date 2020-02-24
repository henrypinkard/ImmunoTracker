import h5py
import numpy as np
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

from util import *

def plot_t_cell_motility():
    sqrt_time = True
    num_bootstraps = 500
    num_query = 20

    ########      T cell motility coefficent at 24 hours     #############
    filepath49 = '/Users/henrypinkard/Desktop/imaris_analysis/49.ims'
    time_cal_filepath49 = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/49_uncorrected.ims'

    elapsed_time_s = read_time_calibration(time_cal_filepath49)
    file49 = h5py.File(filepath49, mode='r')

    coords, time_indices, tracks = get_spots(file49, 'GFP')
    data_gfp = compute_displacements_over_time(coords, tracks, time_indices, elapsed_time_s)
    coords, time_indices, tracks = get_spots(file49, 'RFP')
    data_rfp = compute_displacements_over_time(coords, tracks, time_indices, elapsed_time_s)
    coords, time_indices, tracks = get_spots(file49, 'VPD')
    data_vpd = compute_displacements_over_time(coords, tracks, time_indices, elapsed_time_s)

    # plt.plot((data_vpd[:, 0]), data_vpd[:, 1], '.') #plot raw data

    query_points_gfp = np.linspace(0, np.max(np.sqrt(data_gfp[:, 0]) if sqrt_time else data_rfp[:, 0]), num_query)
    query_points_rfp = np.linspace(0, np.max(np.sqrt(data_rfp[:, 0]) if sqrt_time else data_rfp[:, 0]), num_query)
    query_points_vpd = np.linspace(0, np.max(np.sqrt(data_vpd[:, 0]) if sqrt_time else data_vpd[:, 0]), num_query)

    bootstrap_data_gfp = bootstrap(data_gfp, query_points_gfp, num_bootstraps=num_bootstraps, sqrt=sqrt_time)
    bootstrap_data_rfp = bootstrap(data_rfp, query_points_rfp, num_bootstraps=num_bootstraps, sqrt=sqrt_time)
    bootstrap_data_vpd = bootstrap(data_vpd, query_points_vpd, num_bootstraps=num_bootstraps, sqrt=sqrt_time)


    motility_coeffs_gfp = data_gfp[:, 1] / np.sqrt(data_gfp[:, 0])
    motility_coeffs_rfp = data_rfp[:, 1] / np.sqrt(data_rfp[:, 0])
    motility_coeffs_vpd = data_vpd[:, 1] / np.sqrt(data_vpd[:, 0])
    bins = np.linspace(0, np.max(np.concatenate([motility_coeffs_vpd, motility_coeffs_gfp, motility_coeffs_rfp])), 40)
    plt.figure()
    plt.hist(motility_coeffs_gfp, bins, density=True, histtype='step', fill=False)
    plt.hist(motility_coeffs_rfp, bins, density=True, histtype='step', fill=False)
    plt.hist(motility_coeffs_vpd, bins, density=True, histtype='step', fill=False)
    plt.legend(['GFP', 'RFP', 'VPD'])
    plt.ylabel('Probability density')
    plt.xlabel('Motility coefficent ($\mu$m/$\sqrt{s}$)')
    plt.savefig('figures/T_cell_motility_hists.pdf')

    plt.figure()
    plot_bootstrapped(query_points_gfp, bootstrap_data_gfp)
    plot_bootstrapped(query_points_rfp, bootstrap_data_rfp)
    plot_bootstrapped(query_points_vpd, bootstrap_data_vpd)
    plt.legend(['Polyclonal', 'OT1', 'OT2'])
    plt.xlabel('Square root time ($\sqrt{s}$)')
    plt.ylabel('Displacement ($\mu$m)')
    plt.savefig('figures/t_cell_motility.pdf')
    pass

    # plt.bar(np.arange(len(means)), means, yerr=std_errors, align='center', alpha=0.5, ecolor='black', capsize=10)
    # plt.xticks(np.arange(3), ['Polyclonal', 'OT1', 'OT2'])
    # plt.ylabel('Motility coefficient (um^2 / s)')
    plt.show()
    #

def plot_dc_motility():
    #######      DC motility from 0 to 24 hours     #############
    sqrt_time = True
    num_bootstraps = 5
    num_query = 20
    filepath49 = '/Users/henrypinkard/Desktop/imaris_analysis/49.ims'
    file49 = h5py.File(filepath49, mode='r')
    time_cal_filepath49 = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/49_uncorrected.ims'
    elapsed_time_s49 = read_time_calibration(time_cal_filepath49)

    filepath25 = '/Users/henrypinkard/Desktop/imaris_analysis/25.ims'
    file25 = h5py.File(filepath25, mode='r')
    time_cal_filepath25 = '/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/25_uncorrected.ims'
    elapsed_time_s25 = read_time_calibration(time_cal_filepath25)

    coords, time_indices, tracks = get_spots(file49, 'XCR1')
    data_24hours = compute_displacements_over_time(coords, tracks, time_indices, elapsed_time_s49)
    # coords, time_indices, tracks = get_spots(file25, 'XCR1')
    coords, time_indices, tracks = get_spots(file25, 'XCR1 T cell zone')
    data_control = compute_displacements_over_time(coords, tracks, time_indices, elapsed_time_s25)

    # plt.plot(np.sqrt(data_0hours[:, 0]), data_0hours[:, 1], '.') #plot raw data
    # plt.plot(np.sqrt(data_24hours[:, 0]), data_24hours[:, 1], '.') #plot raw data

    #histograms of motility coefficients
    motility_coeffs_control = data_control[:, 1] / np.sqrt(data_control[:, 0])
    motility_coeffs_24hour = data_24hours[:, 1] / np.sqrt(data_24hours[:, 0])
    bins = np.linspace(0, np.max(np.concatenate([motility_coeffs_control, motility_coeffs_24hour])), 40)
    plt.figure()
    plt.hist(motility_coeffs_control, bins, density=True, histtype='step', fill=False)
    plt.hist(motility_coeffs_24hour, bins, density=True, histtype='step', fill=False)
    plt.legend(['control', '24 hours'])
    plt.ylabel('Probability density')
    plt.xlabel('Motility coefficent ($\mu$m/$\sqrt{s}$)')
    plt.savefig('figures/DC_motility_hists.pdf')


    #LOWESS fit with Bootstrapped error bars
    query_points_0hours = np.linspace(0, np.max(np.sqrt(data_control[:, 0]) if sqrt_time else data_control[:, 0]), num_query)
    query_points_24hours = np.linspace(0, np.max(np.sqrt(data_24hours[:, 0]) if sqrt_time else data_24hours[:, 0]), num_query)
    bootstrap_data_0hours = bootstrap(data_control, query_points_0hours, num_bootstraps=num_bootstraps, sqrt=sqrt_time, alpha=0.4)
    bootstrap_data_24hours = bootstrap(data_24hours, query_points_24hours, num_bootstraps=num_bootstraps, sqrt=sqrt_time, alpha=0.4)
    plt.figure()
    plot_bootstrapped(query_points_0hours, bootstrap_data_0hours)
    plot_bootstrapped(query_points_24hours, bootstrap_data_24hours)
    plt.legend(['control', '24 hours'])
    plt.xlabel('Square root time ($\sqrt{s}$)')
    plt.ylabel('Displacement ($\mu$m)')
    plt.savefig('figures/DC_motility.pdf')
    plt.show()


# plot_t_cell_motility()
plot_dc_motility()
