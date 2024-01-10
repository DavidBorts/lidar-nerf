import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from sortedcontainers import SortedList

def initialize_kalman_filter(delta_t, q, sigma_x, sigma_y, sigma_z, sigma_orient):
    """
    Initializes the Kalman Filter for a passenger car with GNSS/GPS data.

    :param delta_t: Time interval between measurements
    :param q: Process noise parameter
    :param sigma_x: Standard deviation of GPS measurement noise in x-axis
    :param sigma_y: Standard deviation of GPS measurement noise in y-axis
    :param sigma_z: Standard deviation of GPS measurement noise in z-axis
    :return: Initialized matrices and vectors for the Kalman Filter
    """

    # Number of state variables (position, orientation, velocities, turn rates)
    n = 12

    n_measurements = 6  # We measure position and orientation

    # State Transition Matrix (A)
    A = np.eye(n)
    for i in range(6):
        A[i, i + 6] = delta_t

    A[6, 5] = -delta_t  # -v_y influence on x velocity due to yaw
    A[7, 5] = delta_t   # v_x influence on y velocity due to yaw

    # Measurement Matrix (H)
    H = np.zeros((n_measurements, n))
    H[:n_measurements, :n_measurements] = np.eye(n_measurements)

    # Process Noise Covariance (Q)
    # Q = q * np.array([[delta_t**4/4, 0, 0, delta_t**3/2, 0, 0],
    #                   [0, delta_t**4/4, 0, 0, delta_t**3/2, 0],
    #                   [0, 0, delta_t**4/4, 0, 0, delta_t**3/2],
    #                   [delta_t**3/2, 0, 0, delta_t**2, 0, 0],
    #                   [0, delta_t**3/2, 0, 0, delta_t**2, 0],
    #                   [0, 0, delta_t**3/2, 0, 0, delta_t**2]])

    # Process Noise Covariance (Q)
    Q = q * np.array([
        [delta_t ** 4 / 4, 0, 0, delta_t ** 3 / 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, delta_t ** 4 / 4, 0, 0, delta_t ** 3 / 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, delta_t ** 4 / 4, 0, 0, delta_t ** 3 / 2, 0, 0, 0, 0, 0, 0],
        [delta_t ** 3 / 2, 0, 0, delta_t ** 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, delta_t ** 3 / 2, 0, 0, delta_t ** 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, delta_t ** 3 / 2, 0, 0, delta_t ** 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, delta_t ** 3 / 2, 0, 0, 0, 0, 0],  # For velocity terms
        [0, 0, 0, 0, 0, 0, 0, delta_t ** 3 / 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, delta_t ** 3 / 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, delta_t ** 2, 0, 0],  # For angular velocity terms
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, delta_t ** 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, delta_t ** 2]
    ])

    # Measurement Noise Covariance (R)
    R = np.diag([sigma_x**2, sigma_y**2, sigma_z**2]+ [sigma_orient**2] * 3)

    # Initial Estimate Uncertainty (P)
    P = np.eye(n) * 1000


    return A, H, Q, R, P


def find_closest(sorted_list, item):
    """ Find the index of the closest value to the item. """
    pos = sorted_list.bisect_left(item)
    if pos == 0:
        return 0
    if pos == len(sorted_list):
        return len(sorted_list) - 1
    before = sorted_list[pos - 1]
    after = sorted_list[pos]
    if after - item < item - before:
        return pos
    else:
        return pos - 1

class GNSSHandler(object):

    def __init__(self, timestamps, gnss_values, delta_t=0.033, sigma_x=0.2, sigma_y=0.2, sigma_z=0.2, sigma_orient=0.1, q=1000):
        initial_state_mean = [gnss_values[0, 0],
                              gnss_values[0, 1],
                              gnss_values[0, 2],
                              gnss_values[0, 3],
                              gnss_values[0, 4],
                              gnss_values[0, 5],
                              gnss_values[1, 0] - gnss_values[0, 0],
                              gnss_values[1, 1] - gnss_values[0, 1],
                              gnss_values[1, 2] - gnss_values[0, 2],
                              gnss_values[1, 3] - gnss_values[0, 3],
                              gnss_values[1, 4] - gnss_values[0, 4],
                              gnss_values[1, 5] - gnss_values[0, 5]]
        idx = np.argsort(timestamps)
        print(idx)
        timestamps = SortedList(np.array(timestamps)[idx])
        gnss_values = np.array(gnss_values)[idx.tolist()]
        self.timestamps = timestamps
        self.gnss_values = gnss_values
        A,H,Q,R,P = initialize_kalman_filter(delta_t, q, sigma_x, sigma_y, sigma_z, sigma_orient)
        self.kf = KalmanFilter(transition_matrices=A, observation_matrices=H,
                               transition_covariance=Q, observation_covariance=R,
                               initial_state_covariance=P, initial_state_mean=initial_state_mean)
        # initalize values
        self.smoothed_state_means, self.smoothed_state_covariances = self.apply_advanced_kalman(self.gnss_values)

    def find_updated_position(self, timestamp):
        idx = find_closest(self.timestamps, timestamp)
        return self.smoothed_state_means[idx]

    def get_closest_timestamp(self, target_timestamps):
        pass

    def apply_advanced_kalman(self, xyzrpys):
        self.kf = self.kf.em(xyzrpys.copy(), n_iter=15)
        print(xyzrpys.shape)
        (smoothed_state_means, smoothed_state_covariances) = self.kf.smooth(xyzrpys)

        return smoothed_state_means, smoothed_state_covariances

    def kalman_smoothing(self):
        xyzrpys = self.gnss_values
        initial_state_mean = [xyzrpys[0, 0], xyzrpys[1, 0] - xyzrpys[0, 0], xyzrpys[0, 1],
                              xyzrpys[1, 1] - xyzrpys[0, 1],
                              xyzrpys[0, 2], xyzrpys[1, 2] - xyzrpys[0, 2], xyzrpys[0, 3],
                              xyzrpys[1, 3] - xyzrpys[0, 3],
                              xyzrpys[0, 4], xyzrpys[1, 4] - xyzrpys[0, 4], xyzrpys[0, 5],
                              xyzrpys[1, 5] - xyzrpys[0, 5]]
        transition_matrix = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

        observation_matrix = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]

        kf1 = KalmanFilter(transition_matrices=transition_matrix,
                           observation_matrices=observation_matrix,
                           initial_state_mean=initial_state_mean)
        kf1 = kf1.em(xyzrpys.copy(), n_iter=5)
        (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(xyzrpys)
        print(kf1.observation_covariance)
        kf2 = KalmanFilter(transition_matrices=transition_matrix,
                           observation_matrices=observation_matrix,
                           initial_state_mean=initial_state_mean,
                           observation_covariance=np.diag([1, 1, 1, 1, 1, 1]) * 0.1,
                           em_vars=['transition_covariance', 'initial_state_covariance'])
        kf2 = kf2.em(xyzrpys, n_iter=5)
        (smoothed_state_means, smoothed_state_covariances) = kf2.smooth(xyzrpys)
        self.smoothed_state_means = smoothed_state_means

    def visualize_smothed_gnns_data(self):
        # visualizing GNSS frames
        xyzrpys = self.gnss_values


        ax = plt.figure().add_subplot(projection='3d')

        ax.scatter(xyzrpys[:, 0], xyzrpys[:, 1], xyzrpys[:, 2], c=xyzrpys[:, 2])
        ax.scatter(self.smoothed_state_means[:, 0], self.smoothed_state_means[:, 2], self.smoothed_state_means[:, 4], c='r')
        plt.show()

        plt.figure(1)
        times = range(xyzrpys.shape[0])
        plt.plot(  # times, xyzrpys[:, 0], 'bo',
            self.timestamps, xyzrpys[:, 2], 'ro',
            # times, smoothed_state_means[:, 0], 'b--',
            self.timestamps, self.smoothed_state_means[:, 4], 'r--', )
        plt.show()

    def initialze_binary(self):
        pass

    def return_closest_value(self, timestamp):
        pass

    def interpolare_linear_closest_value(self, timestamp):
        pass