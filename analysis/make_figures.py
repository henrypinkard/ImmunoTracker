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


def read_time_calibration(time_cal_filepath):
    # read time calibration
    time_cals = []
    time_file = h5py.File(time_cal_filepath, mode='r')
    for key in time_file['DataSetInfo']['TimeInfo'].attrs.keys():
        if 'TimePoint' in key and 'Dataset' not in key and 'File' not in key:
            string = ''.join([letter.decode("utf-8") for letter in time_file['DataSetInfo']['TimeInfo'].attrs[key]])
            split1 = string.split(' ')[1].split(':')
            h = int(split1[0][-1])
            m = int(split1[1]) % 60
            s = float(split1[2]) % 60
            total_s = h * 60 * 60 + m * 60 + s
            time_cals.append(total_s)
    time_cals.sort()
    elapsed_time_s = np.array(time_cals)
    return elapsed_time_s


def get_spots(file, name):
    for folder_name in file['Scene']['Content']:
        object = file['Scene']['Content'][folder_name]
        object_name = object.attrs['Name'][0].decode('utf-8')
        if object_name == name:
            coords = object['CoordsXYZR'][:, :3]
            time_indices = np.ravel(object['Time'])
            edges = object['Edges']
            return coords, time_indices, edges_to_tracks(edges)


def edges_to_tracks(edges):
    # sort edges into list of list of list of all indices in a given track
    tracks = []
    for edge in edges:
        new_track = True
        for track in tracks:
            if edge[0] in track or edge[1] in track:
                track.add(edge[0])
                track.add(edge[1])
                new_track = False
        if new_track:
            tracks.append(set())
            tracks[-1].add(edge[0])
            tracks[-1].add(edge[1])
    # convert to arrays
    tracks_arrays = []
    for track in tracks:
        tracks_arrays.append(np.array(list(track)))
    return tracks_arrays


def compute_displacements(coords, tracks, time_indices, elapsed_time_s):
    displacements = []
    total_times = []
    for track in tracks:
        ts = time_indices[track]
        min_time = elapsed_time_s[min(ts)]
        max_time = elapsed_time_s[max(ts)]
        total_times.append(max_time - min_time)
        # get xyz indices corresponding to min and max time
        min_index = track[np.argmin(ts)]
        max_index = track[np.argmax(ts)]
        start_xyz = coords[min_index]
        end_xyz = coords[max_index]
        displacements.append(np.linalg.norm(end_xyz - start_xyz, 2))
    d = np.array(displacements)
    t = np.array(total_times)
    m = d ** 2 / (6 * t)
    return t, m


def compute_displacements_over_time(coords, tracks, time_indices, elapsed_time_s):
    time_displacement_tuples = []
    for track in tracks:
        sorted_track = np.sort(track)
        for start_index in range(sorted_track.size - 1):
            for end_index in range(start_index + 1, sorted_track.size):
                start_spot_index = sorted_track[start_index]
                end_spot_index = sorted_track[end_index]
                # num_time_steps = time_indices[end_spot_index] - time_indices[start_spot_index]
                track_time = elapsed_time_s[time_indices[end_spot_index]] - elapsed_time_s[
                    time_indices[start_spot_index]]
                track_displacement = np.linalg.norm(coords[end_spot_index] - coords[start_spot_index], 2)
                time_displacement_tuples.append((track_time, track_displacement))
    return np.array(time_displacement_tuples)


def lowess(x, y, query_points, alpha):
    predicted_vals = []
    for query_point in query_points:
        # find alpha fraction of closest points
        distances = np.abs(x - query_point)
        mask = np.argsort(distances)[:int(alpha * x.size)]
        # train regressor
        A = x[mask][:, None]
        b = y[mask][:, None]

        if np.max(mask) == int(alpha * x.size) - 1:  # Don't have intercept term for first one
            A = np.concatenate([A, np.ones_like(A)], axis=1)  # add bias term

        dist = np.abs((x[mask] - query_point))
        dist /= np.max(dist)
        w = np.sqrt(np.diag((1 - dist ** 3) ** 3))
        # rewight to do weighted least sqaures
        b_w = np.dot(w, b)
        A_w = np.dot(w, A)
        # solve with least squares solver
        coeffs = np.linalg.lstsq(A_w, b_w)[0].T
        if coeffs.size == 1:
            coeffs = np.concatenate([coeffs, np.array([[0]])], axis=1)
        # predict value based on this regressor
        predicted_vals.append(coeffs[:, 0] * query_point + coeffs[:, 1])
    return np.ravel(np.array(predicted_vals))


# Locally weighted regression estiamte. Cleveland, W.S. (1979) “Robust Locally Weighted Regression and Smoothing Scatterplots”. Journal of the American Statistical Association 74 (368): 829-836.
# Bootstrap estimate: https://thestatsgeek.com/2013/07/02/the-miracle-of-the-bootstrap/

def bootstrap(track_data, query_points, num_bootstraps=200, sqrt=True, alpha=0.4):
    sqrt_times = np.sqrt(track_data[:, 0]) if sqrt else track_data[:, 0]
    displacements = track_data[:, 1]
    bootstrap_predctions = []
    for bootstrap_iter in range(num_bootstraps):
        print('Bootstrap iteration {}'.format(bootstrap_iter))
        indices = np.random.randint(0, sqrt_times.shape[0], size=sqrt_times.shape[0])
        sqrt_time_resampled = sqrt_times[indices]
        displacement_resampled = displacements[indices]
        query_pred = lowess(sqrt_time_resampled, displacement_resampled, query_points, alpha=alpha)
        bootstrap_predctions.append(query_pred)
    return np.stack(bootstrap_predctions, axis=0)


def plot_bootstrapped(query_points, bootstrap_data):
    error_lo = np.percentile(bootstrap_data, 2.5, axis=0)
    error_hi = np.percentile(bootstrap_data, 97.5, axis=0)
    mean = np.mean(bootstrap_data, axis=0)
    plt.plot(query_points, mean, '-')
    plt.fill_between(query_points, error_lo, error_hi, alpha=0.4)

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
    num_bootstraps = 500
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


plot_t_cell_motility()
plot_dc_motility()
