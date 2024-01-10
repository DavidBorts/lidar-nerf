import os
from pathlib import Path
import numpy as np
import camtools as ct
import json
import tqdm

from utils import substract_xyz, latlonrpy_to_xyzrpy, ublox_to_gnss2ned, read_gnss_file, match_frames
from kalman import GNSSHandler

class ETHLoader:
    def __init__(self, eth_root, data_path, lidar_dir, gnss_dir) -> None:
        # Root directory.
        self.eth_root = Path(eth_root)
        if not self.eth_root.is_dir():
            raise FileNotFoundError(f"ETH {eth_root} not found.")
        
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise FileNotFoundError(f"ETH {data_path} not found.")
        
        self.lidar_dir = Path(lidar_dir)
        if not self.lidar_dir.is_dir():
            raise FileNotFoundError(f"ETH {lidar_dir} not found.")
        
        self.gnss_dir = Path(gnss_dir)
        if not self.gnss_dir.is_dir():
            raise FileNotFoundError(f"ETH {gnss_dir} not found.")

        # Calibration JSON
        self.calibration_path = self.eth_root / "calib3.json"
        if not self.calibration_path.is_file():
            raise FileNotFoundError(
                f"Calibration json {self.calibration_path} not found."
            )
        
        # reading in LiDAR and GNSS files
        lidar_frames = os.listdir(self.lidar_dir)
        self.lidar_frames = sorted([f for f in lidar_frames if f.endswith('.bin')])
        gnss_frames = os.listdir(self.gnss_dir)
        self.gnss_frames = sorted([f for f in gnss_frames if f.endswith('.txt')])
        
    def load_calibrations(self):
        '''
        ### Read in extrinsic matrices from given calibration file

        arguments:
            :param path2calib: pathlib.Path to JSON calibration file
            :param calib_type: int in [1, 2, 3], calibration format
        '''
        with open(self.calibration_path,) as f:
            calibs = json.load(f)["extrinsics"]

        lidar2gnss = np.array(calibs["lidar2gnss"], dtype=np.float32)
        radar2gnss = np.array(calibs["radar2gnss"], dtype=np.float32)
        return lidar2gnss, radar2gnss

    def _load_all_lidars(self, sequence_name):
        """
        Args:
            sequence_name: str, name of sequence. e.g. "2013_05_28_drive_0000".

        Returns:
            velo_to_world: 4x4 metric.
        """

        # dict to store results
        lidar2world_dict = dict()

        # match lidar frames to closest GNSS frames, by timestamp
        _, matched_gnss_frames = match_frames(self.lidar_frames, self.gnss_frames)
        assert(len(matched_gnss_frames) == len(self.lidar_frames))

        # load in calibrations
        lidar2gnss, _ = self.load_calibrations()

        # Read in GNSS data and compute world coords
        zero_position = None
        zero_position_xyzrpy = None
        timestamp_zero = None
        timestamps = list()
        xyzrpys = []
        for gnss_frame in matched_gnss_frames:

            # Reading GNSS data -> x,y,z,roll,pitch,yaw
            data = read_gnss_file(self.gnss_dir / gnss_frame)

            # keeping track of first timestamp in sequence
            timestamp = int(gnss_frame.split('.')[0])
            if timestamp_zero is None:
                timestamp_zero = timestamp
            timestamp = timestamp - timestamp_zero
            timestamps.append(timestamp)

            xyzrpy = latlonrpy_to_xyzrpy(data['longitude'],
                                         data["latitude"],
                                         data["height"],
                                         data["roll"],
                                         data["pitch"],
                                         data["yaw"])
            assert data["yaw"] == data["heading"], 'heading and yaw are not equal'
            assert data["lon"] == data["longitude"], 'longitude and lon are not equal'
            assert data["lat"] == data["latitude"], 'latitude and lat are not equal'

            # centering position on first frame
            if zero_position_xyzrpy is None:
                zero_position_xyzrpy = xyzrpy
            xyzrpy = substract_xyz(zero_position_xyzrpy, xyzrpy)
            xyzrpys.append(xyzrpy)
            if zero_position is None:
                zero_position = data

        # applying kalman filter on world coords
        xyzrpys = np.array(xyzrpys)
        gnss = GNSSHandler(timestamps, xyzrpys)
        smoothed_state_means, smoothed_state_covariances = gnss.apply_advanced_kalman(xyzrpys)
        
        # compute gnss2world and lidar2world matrices
        for gnss_frame, lidar_frame in tqdm.tqdm(zip(matched_gnss_frames, self.lidar_frames), desc=f"Computing extrinsic matrices"):

            # read in txt data for each gnss frame
            data = read_gnss_file(self.gnss_dir / gnss_frame)

            # logging (centered) timestamp
            timestamp = int(gnss_frame.split('.')[0])
            timestamp = timestamp - timestamp_zero

            # converting gnss coordinates to world xyzrpy coords
            xyzrpy = latlonrpy_to_xyzrpy(data['longitude'], 
                                        data["latitude"], 
                                        data["height"], 
                                        data["roll"], 
                                        data["pitch"], 
                                        data["yaw"])

            # centering on first frame
            xyzrpy = substract_xyz(zero_position_xyzrpy, xyzrpy)

            # applying kalman filter
            xyzrpy = gnss.find_updated_position(timestamp)

            # compute gnss2world (offset by origin)
            data = {
                "longitude": xyzrpy[0],
                "latitude": xyzrpy[1],
                "height": xyzrpy[2],
                "roll": xyzrpy[3],
                "pitch": xyzrpy[4],
                "yaw": xyzrpy[5]
            }

            # computing gnss2world
            gnss2world = ublox_to_gnss2ned(data, reference_point_ublox_data=zero_position)

            # use gnss2world to compute lidar2world for this frame
            lidar2world = np.matmul(gnss2world, lidar2gnss)
            lidar2world_dict[lidar_frame] = lidar2world

        raise NotImplementedError()
        data_poses_dir = self.data_poses_dir / f"{sequence_name}_sync"
        assert data_poses_dir.is_dir()

        # IMU to world transformation (poses.txt).
        poses_path = data_poses_dir / "poses.txt"
        imu_to_world_dict = dict()
        frame_ids = []
        for line in np.loadtxt(poses_path):
            frame_id = int(line[0])
            frame_ids.append(frame_id)
            imu_to_world = line[1:].reshape((3, 4))
            imu_to_world_dict[frame_id] = imu_to_world

        # Camera to IMU transformation (calib_cam_to_pose.txt).
        cam_to_imu_path = self.calibration_dir / "calib_cam_to_pose.txt"
        with open(cam_to_imu_path, "r") as fid:
            cam_00_to_imu = KITTI360Loader._read_variable(fid, "image_00", 3, 4)
            cam_00_to_imu = ct.convert.pad_0001(cam_00_to_imu)

        # Camera00 to Velo transformation (calib_cam_to_velo.txt).
        cam00_to_velo_path = self.calibration_dir / "calib_cam_to_velo.txt"
        with open(cam00_to_velo_path, "r") as fid:
            line = fid.readline().split()
            line = [float(x) for x in line]
            cam_00_to_velo = np.array(line).reshape(3, 4)
            cam_00_to_velo = ct.convert.pad_0001(cam_00_to_velo)

        # Compute velo_to_world
        velo_to_world_dict = dict()
        for frame_id in frame_ids:
            imu_to_world = imu_to_world_dict[frame_id]
            cam_00_to_world_unrec = imu_to_world @ cam_00_to_imu
            velo_to_world = cam_00_to_world_unrec @ np.linalg.inv(cam_00_to_velo)
            velo_to_world_dict[frame_id] = ct.convert.pad_0001(velo_to_world)

        return velo_to_world_dict

    def load_lidars(self, sequence_name, train_frame_ids, test_frame_ids):
        """
        Args:
            sequence_name: str, name of sequence. e.g. "2013_05_28_drive_0000".
            frame_ids: list of int, frame ids. e.g. range(1908, 1971+1).

        Returns:
            velo_to_worlds
        """
        lidar2world_dict = self._load_all_lidars(sequence_name)
        lidar2worlds_train = [lidar2world_dict[frame_id] for frame_id in train_frame_ids]
        lidar2worlds_train = np.stack(lidar2worlds_train)
        lidar2worlds_test = [lidar2world_dict[frame_id] for frame_id in test_frame_ids]
        lidar2worlds_test = np.stack(lidar2worlds_test)

        lidar2worlds = {
            "train": lidar2worlds_train,
            "test": lidar2worlds_test,
            "val": lidar2worlds_test
        }
        return lidar2worlds
