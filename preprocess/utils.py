from copy import copy

import utm
import numpy as np

def latlon_to_utmcm(latitude, longitude):
    '''
    ### Convert latitude and longitude to UTM zone, easting, northing and hemisphere
    '''
    easting, northing, zone_number, zone_letter = utm.from_latlon(latitude, longitude)
    return easting, northing, f"{zone_number}{zone_letter}"

def latlon_to_ned(ublox_height, ublox_lat, ublox_lon):
    """
    ### Converts the latitude longitude and height from GNSS to NED (North-East-Down) coordinates

    Arguments
        :param ublox_height: The height above sea level in meters as measured by the u-blox receiver.
        :param ublox_lat: The latitude in decimal degrees as measured by the u-blox receiver.
        :param ublox_lon: The longitude in decimal degrees as measured by the u-blox receiver.
    Returns
        :return: The geographic coordinates converted to North East Down (NED) format in meters.
    """
    easting, northing, utm_zone = latlon_to_utmcm(ublox_lat, ublox_lon)
    down = -ublox_height
    gps_point_meters = np.array([northing, easting, down]).reshape((3, 1))
    return gps_point_meters

def latlonrpy_to_xyzrpy(longitude, latitude, height, roll, pitch, yaw):
    '''
    ### Converts GNSS latitude, longitude, roll, pitch, and yaw readings
    ### into x, y, z, roll, pitch, and yaw, where roll/pitch/yaw are in
    ### radians, not degrees
    '''

    ublox_lon = longitude / 1e7  # Convert from deg * 1e-7 to deg
    ublox_lat = latitude / 1e7  # Convert from deg * 1e-7 to deg
    ublox_height = height / 1e3  # Convert from mm to meters
    ned = latlon_to_ned(ublox_height, ublox_lat, ublox_lon)

    ublox_roll = np.deg2rad(roll / 1e5)
    ublox_pitch = np.deg2rad(pitch / 1e5)
    ublox_yaw = np.deg2rad(yaw / 1e5)
    rpy = np.array([ublox_roll, ublox_pitch, ublox_yaw])

    xyzrpy = [float(v) for v in ned] + [float(v) for v in rpy]
    return xyzrpy

def get_translation_gnss2ned(ublox_data, reference_point_ublox_data=None):
    """
    ### Calculates the translation vector from GNSS coordinates to NED (North-East-Down) coordinates.

    Arguments
        :param ublox_data: the GNSS coordinates (lon, lat, height) in u-blox format
        :param reference_point_ublox_data: the reference GNSS coordinates (lon, lat, height) in u-blox format (default=None)
    Returns
        :return: the translation vector from GNSS coordinates to NED coordinates

    """
    if reference_point_ublox_data is None:
        reference_point = np.zeros((3, 1))
    else:
        reference_point_lon = reference_point_ublox_data['longitude'] / 1e7  # Convert from deg * 1e-7 to deg
        reference_point_lat = reference_point_ublox_data['latitude'] / 1e7  # Convert from deg * 1e-7 to deg
        reference_point_height = reference_point_ublox_data['height'] / 1e3  # Convert from mm to meters
        reference_point = latlon_to_ned(reference_point_height, reference_point_lat, reference_point_lon)

    ublox_lon = ublox_data['longitude'] / 1e7  # Convert from deg * 1e-7 to deg
    ublox_lat = ublox_data['latitude'] / 1e7  # Convert from deg * 1e-7 to deg
    ublox_height = ublox_data['height'] / 1e3  # Convert from mm to meters
    gps_point_meters = latlon_to_ned(ublox_height, ublox_lat, ublox_lon)

    # Deduct reference point
    print(ublox_data)
    print(f"orig point: {gps_point_meters}")
    print(f"ref point: {reference_point}")
    gps_point_meters_relative = gps_point_meters - reference_point

    return gps_point_meters_relative.squeeze()

def get_rotation_gnss2ned(roll, pitch, heading):
    """
    ### This method calculates the rotation matrix from GNSS (Global Navigation Satellite System)
    ### to NED (North-East-Down) coordinates. It takes in the roll, pitch, and heading angles in degrees
    ### and returns the rotation matrix using ZYX euler angles.

    Arguments:
        :param roll: The roll angle in degrees.
        :param pitch: The pitch angle in degrees.
        :param heading: The heading angle in degrees.
    Returns:
        :return: The rotation matrix from GNSS to NED coordinates.
    """

    # converting to radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    heading = np.deg2rad(heading)

    #return R.from_euler('zyx', [heading, pitch, roll]).as_matrix()

    C_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), np.sin(roll)],
                    [0, -np.sin(roll), np.cos(roll)]])

    C_y = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                    [0, 1, 0],
                    [np.sin(pitch), 0, np.cos(pitch)]])

    C_z = np.array([[np.cos(heading), np.sin(heading), 0],
                    [-np.sin(heading), np.cos(heading), 0],
                    [0, 0, 1]])

    rotation_gnss2ned = C_z.T.dot(C_y.T).dot(C_x.T)
    return rotation_gnss2ned

def ublox_to_gnss2ned(ublox_data, reference_point_ublox_data=None):
    """
    ### Convert ublox data to GNSS to NED (North-East-Down) transformation matrix.

    Arguments:
        :param ublox_data: A dictionary containing 'roll', 'pitch', and 'heading' keys with corresponding values in deg * 1e-5 format.
        :param reference_point_ublox_data: [OPTIONAL] dictionary containing 'roll', 'pitch', and 'heading' keys with corresponding values in deg * 1e-5 format.\
    Returns:
        :return: A 4x4 transformation matrix representing the GNSS to NED transformation.
    """
    ublox_roll = ublox_data['roll'] / 1e5  # Convert from deg * 1e-5 to deg
    ublox_pitch = ublox_data['pitch'] / 1e5  # Convert from deg * 1e-5 to deg
    ublox_heading = ublox_data['yaw'] / 1e5  # Convert from deg * 1e-5 to deg

    # Get rotation matrix & translation vector
    rotation_gnss2ned = get_rotation_gnss2ned(ublox_roll, ublox_pitch, ublox_heading) # GPS_to_NED
    translation_gnss2ned = get_translation_gnss2ned(ublox_data, reference_point_ublox_data)

    # Embedding the rotation matrix in a homogeneous transformation matrix
    gnss2ned = np.eye(4)
    gnss2ned[:3, :3] = rotation_gnss2ned
    gnss2ned[:3, 3] = translation_gnss2ned
    return gnss2ned

def substract_xyz(vec1, vec2):
    '''
    ### Subtract vec1[0:3] from vec2[0:3], regardless of len(vec1) or len(vec2).

    NOTE: vec1 and vec2 must be one-dimensional.
    '''

    diff = copy(vec2)
    diff[0] = vec2[0] - vec1[0]
    diff[1] = vec2[1] - vec1[1]
    diff[2] = vec2[2] - vec1[2]

    return diff

def match_frames(source_frames, target_frames):
    '''
    '''

    # capturing source and target file extensions
    source_ext = "." + source_frames[0].split('.')[1]
    target_ext = "." + target_frames[0].split('.')[1]

    # since all filenames are nanosecond timestamps, reformat
    # as ndarrays of ints
    source = [int(x.split('.')[0]) for x in source_frames]
    source = np.array(source)
    target = [int(x.split('.')[0]) for x in target_frames]
    target = np.array(target)

    # find closest timestamps by computing difference matrix
    diffs = np.abs(source[..., None] - target[None, ...])
    inds = np.argmin(diffs,-1)
    target_matches = target[inds]

    source = [str(x) + source_ext for x in source.tolist()]
    target_matches = [str(x) + target_ext for x in target_matches.tolist()]
    return source, target_matches

def read_gnss_file(path):
    """
    ### Read in GNSS data from .txt file

    Arguments:
        :param path: pathlib.Path to GNSS .txt file
    """
    out_data = dict()

    """
    GNSS format:

    iTOW                   # GPS Millisecond time of week [ms]
    timestamp              # UTC timestamp in microseconds  [us]
    tAcc                   # time accuracy estimate [ns] (UTC)
    lon                    # Longitude [deg / 1e-7]
    lat                    # Latitude [deg / 1e-7]
    height                 # Height above Ellipsoid [mm]
    hMSL                   # Height above mean sea level [mm]
    fixType                # GNSS fix Type, range 0..5
    numSV                  # Number of SVs used in Nav Solution
    hAcc                   # Horizontal Accuracy Estimate [mm]
    vAcc                   # Vertical Accuracy Estimate [mm]
    roll                   # Vehicle roll. [deg / 1e-5]
    pitch                  # Vehicle pitch. [deg / 1e-5]
    heading                # Vehicle heading. [deg / 1e-5]
    accRoll                # Vehicle roll accuracy (if null, roll angle is not available). [deg / 1e-5]
    accPitch               # Vehicle pitch accuracy (if null, pitch angle is not available). [deg / 1e-5]
    accHeading             #Vehicle heading accuracy [deg / 1e-5]
    angular_rate_roll      # Angular rate around roll axis [(deg / s) / 1e-5]
    angular_rate_pitch     # Angular rate around pitch axis [(deg / s) / 1e-5]
    angular_rate_heading   # Angular rate of heading (around the negative yaw axis) [(deg / s) / 1e-5]
    velN                   # NED north velocity [mm/s]
    velE                   # NED east velocity [mm/s]
    velD                   # NED down velocity [mm/s]
    gSpeed                 # Ground Speed (2-D) [mm/s]. Can be negative.
    sAcc                   # Speed Accuracy Estimate [mm/s]
    pDOP                   # Position DOP [1 / 0.01]
    magDec                 # Magnetic declination [deg / 1e-2]
    magAcc                 # Magnetic declination accuracy [deg / 1e-2]
    """
    with open(path, 'r') as f:
        data = f.readlines()[0].split(' ')
        out_data['timestamp'] = float(data[1])
        out_data['longitude'] = float(data[3])
        out_data['latitude'] = float(data[4])
        out_data['lon'] = float(data[3])
        out_data['lat'] = float(data[4])
        out_data['height'] = float(data[6])
        out_data['roll'] = float(data[11])
        out_data['pitch'] = float(data[12])
        out_data['yaw'] = float(data[13])
        out_data['heading'] = float(data[13])

    return out_data