import numpy as np
from scipy.spatial import Delaunay


class LAMI_helper:
    """"
    This class provides convenience functions for computing surface interpolation and computing physics-based features
    for these interpolations to feed into the neural net. It corresponds to a fixed set of interpolation points
    on the sample surface. If a new set of points is available, a new instance of the calss should be created
    """

    def __init__(self, interpolation_points):
        """

        :param interpolation_points: N x 3 numpy array of XYZ points
        """
        self.tris = Delaunay(interpolation_points[:, :2])
        self.simplices = self.tris.simplices
        self.interp_points = interpolation_points


    def bin_surface_distance(self, dist, binmax=350, numbins=12, azimuth_indices=[2]):
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


    def get_interp_val(self, xy):
        indices = self.tris.find_simplex(xy)
        if -1 in indices:
            return None
        vertex_indices = self.tris.simplices[self.tris.find_simplex(xy)]
        vertices = self.interp_points[vertex_indices]

        edge1 = vertices[1] - vertices[0]
        edge2 = vertices[2] - vertices[1]
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        # n dot (x - x0) = 0, solve for z coordinate
        z_val = np.sum((xy - vertices[0, :2]) * normal[:2]) / -normal[2] + vertices[0, 2]
        return z_val

    def is_under_surface(self, xyz):
        z = self.get_interp_val(xyz[:2])
        if z is None:
            return False
        return xyz[2] < z

    def binary_search(self, initial_point, direction, min_distance, max_distance, search_tol=0.1):
        half_dist = (min_distance + max_distance) / 2.0
        if (max_distance - min_distance < search_tol):
            return half_dist

        search_point = initial_point + direction * half_dist
        if not self.is_under_surface(search_point):
            return self.binary_search(initial_point, direction, min_distance, half_dist)
        else:
            return self.binary_search(initial_point, direction, half_dist, max_distance)

    def compute_dist_to_interp(self, initial_point, direction_unit_vec, search_start_dist=400):
        initial_dist = search_start_dist
        # start with a point outside and then binary line search for the distance
        while self.is_under_surface(initial_point + direction_unit_vec * initial_dist):
            initial_dist = initial_dist * 2

        return self.binary_search(initial_point, direction_unit_vec, 0, initial_dist)

    def compute_distances(self, initial_point, n_samples_theta=12, n_samples_phi=5, phi_max_deg=50):
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
                                   self.compute_dist_to_interp(initial_point, direction),
                             axis=2, arr=direction_vecs)[None, :, :]

    def construct_vector(self, xyz, brightness, fov_position):
        distances = self.compute_distances(xyz)
        dist_hist = self.bin_surface_distance(distances)
        feature_vec = np.concatenate([dist_hist, [[*fov_position, brightness]]], axis=1)
        return feature_vec

    # def predict_power(model, data):
    #     voltage_to_power = lambda x: (np.cos(3.1415 + 2 * 3.1415 / 510 * x) + 1) / 2
    #     predicted_voltage = model.predict(data)
    #     return voltage_to_power(predicted_voltage)
