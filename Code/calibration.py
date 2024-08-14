import numpy as np


class LiDAR2Camera(object):
    """ Calibration object that contains calibration (transformation & rotation) matrices """
    """ NOTE: For calibration diretory that contains the only 1 file 'xxxxxx.txt' """

    def __init__(self, calib_file):
        calibs = self.read_calib_file(calib_file)
        
        # Projection from 3D world coordinates to 2D image coordinates for camera 2
        self.P = calibs["P2"].reshape(3, 4)

        # Rigid transform from Lidar coord to reference camera coord
        self.V2C = calibs["Tr_velo_to_cam"].reshape(3, 4)

        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs["R0_rect"].reshape(3, 3)


    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """

        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    
    
    def convert_3D_to_2D(self, pts_3d_velo, print_info=False):
        """
        Input: 3D Points in LiDAR Coordinates
        Output: 2D Pixels in Image Coordinates
        """

        if print_info:
            print("\nConverting 3D LiDAR Points to 2D Image Pixels ...")

        # Convert R0 to a 4x4 homogeneous transformation matrix
        R0_homo = np.eye(4)
        R0_homo[:3, :3] = self.R0

        # Convert V2C to a 4x4 homogeneous transformation matrix
        V2C_homo = np.eye(4)
        V2C_homo[:3, :4] = self.V2C

        # Compute the full transformation matrix from Velodyne to camera
        transform_matrix = np.dot(R0_homo, V2C_homo)

        # Compute the projection matrix from Velodyne to image
        P_velo_to_img = np.dot(self.P, transform_matrix)

        # Convert the 3D points to homogeneous coordinates
        pts_3d_homo = np.hstack((pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))))

        # Project the points to 2D
        pts_2d_homo = np.dot(P_velo_to_img, pts_3d_homo.T).T

        # Normalize the points
        pts_2d_homo[:, 0] /= pts_2d_homo[:, 2]
        pts_2d_homo[:, 1] /= pts_2d_homo[:, 2]

        # Extract the 2D points
        pts_2d = pts_2d_homo[:, :2]

        return pts_2d


    def convert_3D_to_camera_coords(self, pts_3d_velo, print_info=False):
        """
        Input: 3D Points in LiDAR Coordinates
        Output: 3D Points in Camera Coordinates
        """

        if print_info:
            print("\nConverting 3D LiDAR Points to 3D Camera Points ...")

        # Convert V2C to a 4x4 homogeneous transformation matrix
        V2C_homo = np.eye(4)
        V2C_homo[:3, :4] = self.V2C

        # Convert the 3D points to homogeneous coordinates
        pts_3d_homo = np.hstack((pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))))

        # Transform the points to the camera coordinate system
        pts_3d_cam = np.dot(V2C_homo, pts_3d_homo.T).T

        # Apply the rectification matrix R0
        pts_3d_rect = np.dot(self.R0, pts_3d_cam[:, :3].T).T

        return pts_3d_rect


    def convert_single_3D_to_camera_coords(self, pt_3d_velo, print_info=False):
        """
        Input: Single 3D Point in LiDAR Coordinates
        Output: Single 3D Point in Camera Coordinates
        """

        if print_info:
            print("\nConverting 3D LiDAR Point to 3D Camera Point ...")

        # Convert V2C to a 4x4 homogeneous transformation matrix
        V2C_homo = np.eye(4)
        V2C_homo[:3, :4] = self.V2C

        # Convert the 3D point to homogeneous coordinates
        pt_3d_homo = np.hstack((pt_3d_velo, 1))

        # Transform the point to the camera coordinate system
        pt_3d_cam = np.dot(V2C_homo, pt_3d_homo)

        # Apply the rectification matrix R0
        pt_3d_rect = np.dot(self.R0, pt_3d_cam[:3])

        return pt_3d_rect


    def project_3D_to_2D(self, pt_3d_camera, print_info=False):
        """
        Input: 3D Points in Camera Coordinates
        Output: 2D Pixels in Image Coordinates
        """
        
        if print_info:
            print("\Converting 3D Camera Points to 2D Image Pixels ...")

        # Convert the 3D point to homogeneous coordinates
        pt_3d_homo = np.hstack((pt_3d_camera, 1))

        # Apply the projection matrix
        pt_2d_homo = np.dot(self.P, pt_3d_homo)

        # Normalize the coordinates
        pt_2d = pt_2d_homo[:2] / pt_2d_homo[2]

        return pt_2d
    
    
    def convert_3D_camera_to_LiDAR_coords(self, pts_3d_camera_list, print_info=False):
        """
        Input: 3D Points in Camera Coordinates
        Output: 3D Points in LiDAR Coordinates
        """
        if print_info:
            print("\nConverting 3D Camera Points to 3D LiDAR Points ...")

        # Invert the rectification matrix R0
        R0_inv = np.linalg.inv(self.R0)

        # Convert V2C to a 4x4 homogeneous transformation matrix
        V2C_homo = np.eye(4)
        V2C_homo[:3, :4] = self.V2C

        # Invert the V2C matrix to transform from camera to LiDAR coordinates
        V2C_inv = np.linalg.inv(V2C_homo)

        # Process each set of 3D points
        pts_3d_velo_list = []
        for pts_3d_camera in pts_3d_camera_list:
            # Rectify camera coordinates
            pts_3d_cam_rect_inv = np.dot(R0_inv, pts_3d_camera.T).T

            # Convert to homogeneous coordinates
            pts_3d_cam_rect_inv_homo = np.hstack((pts_3d_cam_rect_inv, np.ones((pts_3d_cam_rect_inv.shape[0], 1))))

            # Transform back to LiDAR coordinates
            pts_3d_velo = np.dot(V2C_inv, pts_3d_cam_rect_inv_homo.T).T
            pts_3d_velo = pts_3d_velo[:, :3]

            pts_3d_velo_list.append(pts_3d_velo)

        return pts_3d_velo_list
    



class LiDAR2Camera_KITTI_raw_data(object):
    """ Calibration object that contains calibration (transformation & rotation) matrices """
    """ NOTE: For calibration diretory that contains the 3 files: 'calib_cam_to_cam.txt' and 'calib_velo_to_cam.txt' """

    def __init__(self, c2c_calib_file, v2c_calib_file):
        calibs_cam_to_cam = self.read_calib_file(c2c_calib_file)
        calibs_lidar_to_cam = self.read_calib_file(v2c_calib_file)
        
        # Projection from 3D world coordinates to 2D image coordinates for camera 2
        self.P = calibs_cam_to_cam["P_rect_02"].reshape(3, 4)

        # Rigid transform from Lidar coord to reference camera coord
        # Rotation matrix R
        R = calibs_lidar_to_cam["R"].reshape(3, 3)
        # Translation vector T
        T = calibs_lidar_to_cam["T"].reshape(1, 3)
        # Concatenate R and T to form the transformation matrix Tr_velo_to_cam
        self.V2C = np.zeros((3, 4))
        self.V2C[:, :3] = R
        self.V2C[:, 3] = T

        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs_cam_to_cam["R_rect_00"].reshape(3, 3)


    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """

        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    
    
    def convert_3D_to_2D(self, pts_3d_velo, print_info=False):
        """
        Input: 3D Points in LiDAR Coordinates
        Output: 2D Pixels in Image Coordinates
        """

        if print_info:
            print("\nConverting 3D LiDAR Points to 2D Image Pixels ...")

        # Convert R0 to a 4x4 homogeneous transformation matrix
        R0_homo = np.eye(4)
        R0_homo[:3, :3] = self.R0

        # Convert V2C to a 4x4 homogeneous transformation matrix
        V2C_homo = np.eye(4)
        V2C_homo[:3, :4] = self.V2C

        # Compute the full transformation matrix from Velodyne to camera
        transform_matrix = np.dot(R0_homo, V2C_homo)

        # Compute the projection matrix from Velodyne to image
        P_velo_to_img = np.dot(self.P, transform_matrix)

        # Convert the 3D points to homogeneous coordinates
        pts_3d_homo = np.hstack((pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))))

        # Project the points to 2D
        pts_2d_homo = np.dot(P_velo_to_img, pts_3d_homo.T).T

        # Normalize the points
        pts_2d_homo[:, 0] /= pts_2d_homo[:, 2]
        pts_2d_homo[:, 1] /= pts_2d_homo[:, 2]

        # Extract the 2D points
        pts_2d = pts_2d_homo[:, :2]

        return pts_2d
    
    
    def convert_3D_to_camera_coords(self, pts_3d_velo, print_info=False):
        """
        Input: 3D Points in LiDAR Coordinates
        Output: 3D Points in Camera Coordinates
        """

        if print_info:
            print("\nConverting 3D LiDAR Points to 3D Camera Points ...")

        # Convert V2C to a 4x4 homogeneous transformation matrix
        V2C_homo = np.eye(4)
        V2C_homo[:3, :4] = self.V2C

        # Convert the 3D points to homogeneous coordinates
        pts_3d_homo = np.hstack((pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))))

        # Transform the points to the camera coordinate system
        pts_3d_cam = np.dot(V2C_homo, pts_3d_homo.T).T

        # Apply the rectification matrix R0
        pts_3d_rect = np.dot(self.R0, pts_3d_cam[:, :3].T).T

        return pts_3d_rect