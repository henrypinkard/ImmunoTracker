import numpy as np

def bin_surface_distance(dist, binmax=350, numbins=12, azimuth_indices=[2]):
    """

    :param dist: N examples x theta (rotation angle) x phi (azimuthal angle i.e. angle with optical axis)
    :param binmax: bins go from 0 to this value. Should be set to the distance beyond which you cant see anything at max power
    :param numbins: number of bins to use
    :param azimuth_index: which azimuth angles to include in design matrix
    :return:
    """
    # Start with vertical distance
    hist_design_mat = np.zeros((dist.shape[0], 0))

    # bin remaining phi angles into histogram
    for azimuth_index in azimuth_indices:
        binedges = np.power(np.linspace(0, 1, numbins + 1), 1.5) * binmax
        distancesforcurrentphi = dist[:, :, azimuth_index]
        counts = np.apply_along_axis(lambda x: np.histogram(x, binedges)[0], 1, distancesforcurrentphi)
        hist_design_mat = np.concatenate((hist_design_mat, counts), axis=1)

    return hist_design_mat


def get_interp_val(xy, tris, interp_points):
    indices = tris.find_simplex(xy)
    if -1 in indices:
        return None
    vertex_indices = tris.simplices[tris.find_simplex(xy)]
    vertices = interp_points[vertex_indices]

    edge1 = vertices[1] - vertices[0]
    edge2 = vertices[2] - vertices[1]
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)
    # n dot (x - x0) = 0, solve for z coordinate
    z_val = np.sum((xy - vertices[0, :2]) * normal[:2]) / -normal[2] + vertices[0, 2]
    return z_val

def is_under_surface(xyz, tris, interp_points):
    z = get_interp_val(xyz[:2], tris, interp_points)
    if z is None:
        return False
    return xyz[2] < z

def binary_search(initial_point, direction, min_distance, max_distance, tris, interp_points, search_tol=0.1):
    half_dist = (min_distance + max_distance) / 2.0
    if (max_distance - min_distance < search_tol):
        return half_dist

    search_point = initial_point + direction * half_dist
    if not is_under_surface(search_point, tris, interp_points):
        return binary_search(initial_point, direction, min_distance, half_dist, tris, interp_points)
    else:
        return binary_search(initial_point, direction, half_dist, max_distance, tris, interp_points)

def compute_dist_to_interp(initial_point, direction_unit_vec, tris, interp_points, search_start_dist=400):
    initial_dist = search_start_dist
    # start with a point outside and then binary line search for the distance
    while is_under_surface(initial_point + direction_unit_vec * initial_dist, tris, interp_points):
        initial_dist = initial_dist * 2;

    return binary_search(initial_point, direction_unit_vec, 0, initial_dist, tris, interp_points)

def compute_distances(initial_point, tris, interp_points, n_samples_theta=12, n_samples_phi=5, phi_max_deg=50):
    """
    Compute the distances from point to cortex along a grid of phi and theta values

    :param n_samples_theta:
    :param n_samples_phi:
    :param phi_max_deg: maximum angle of the numerical aperture in degrees
    :return:
    """
    thetas = np.linspace(0, 2*np.pi, n_samples_theta)
    phis = np.linspace(0, phi_max_deg / 360.0 * np.pi * 2.0, n_samples_phi)

    phi_grid, theta_grid = np.meshgrid(phis, thetas)
    direction_vecs = np.stack([np.cos(theta_grid) * np.sin(phi_grid),
                               np.sin(theta_grid) * np.sin(phi_grid), np.cos(phi_grid)], axis=2)

    return np.apply_along_axis(lambda direction:
                               compute_dist_to_interp(initial_point, direction, tris, interp_points),
                         axis=2, arr=direction_vecs)[None, :, :]

def construct_vector(xyz, tris, interp_points, brightness, fov_position):
    distances = compute_distances(xyz, tris, interp_points)
    dist_hist = bin_surface_distance(distances)
    feature_vec = np.concatenate([dist_hist, [[*fov_position, brightness]]], axis=1)
    return feature_vec

def predict_power(model, data):
    voltage_to_power = lambda x: (np.cos(3.1415 + 2 * 3.1415 / 510 * x) + 1) / 2
    predicted_voltage = model.predict(data)
    return voltage_to_power(predicted_voltage)

def make_depth_profile(model, xy, interp_points, tris, max_depth=300, n_points=20, from_surface_top=True):
    """
    Make a depth profile of predicted excitation along z for a given xy position
    :param max_depth: depth from top of surface
    :param n_points: number of poitns to sample in profile
    :return:
    """
    z_val = get_interp_val(xy, tris, interp_points)
    if z_val is None:
        return None
    if not from_surface_top:
        z_val = 0
    z_queries = np.linspace(z_val, z_val - max_depth, n_points)
    xyz_query_points = np.concatenate([n_points * [xy], z_queries[:, None]], axis=1)

    feature_vectors = [construct_vector(point, tris, interp_points, fov_position=[0.5, 0.5], brightness=-0.5)
                       for point in xyz_query_points]
    power_predictions = predict_power(model, np.concatenate(feature_vectors, axis=0))
    return power_predictions

def make_central_profile(model, interp_points, tris, square_size=5, num_samples=1, max_depth=300, n_points=20):
    """
    Make a central reference one by averaging a few that are nearby each other
    """
    # center_position = interp_points[np.argmin(interp_points[:, 2])][:2] # take top
    center_position = ((np.max(interp_points, axis=0) - np.min(interp_points, axis=0)) / 2
                                    + np.min(interp_points, axis=0))[:2]
    central_depth_profiles = []
    for i, x in enumerate(
            np.linspace(center_position[0] - square_size, center_position[0] + square_size, num_samples)):
        print('z position profile {} of {}\r'.format(i, num_samples), end='')
        for y in np.linspace(center_position[1] - square_size, center_position[1] + square_size, num_samples):
            central_depth_profiles.append(make_depth_profile(model, [x, y], interp_points, tris, max_depth, n_points))

    return np.squeeze(np.mean(np.stack(central_depth_profiles), axis=0))
