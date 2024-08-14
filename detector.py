import numpy as np
from ultralytics import YOLO
from fusion import *
from utils import *


class YOLOv8Detector:
    def __init__(self, model_path, tracking=False, PCA=False):
        self.model = YOLO(model_path)
        self.tracking = tracking
        self.pca = PCA
        self.last_ground_center_of_id = {}    

    def process_frame(self, frame, pts, lidar2camera, erosion_factor, depth_factor):
        if self.tracking:
            results = self.model.track(
                source=frame,
                classes=[0, 1, 2, 3, 5, 6, 7],
                verbose=False,
                show=False,
                persist=True,
                tracker='bytetrack.yaml'
            )
        else:
            results = self.model.predict(
                source=frame,
                classes=[0, 1, 2, 3, 5, 6, 7],
                verbose=False,
                show=False,
            )

        # Get the results from the YOLOv8-seg model
        r = results[0]
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs

        # Preprocess LiDAR point cloud
        points = np.fromfile(pts, dtype=np.float32).reshape((-1, 4))[:, 0:3]
        point_cloud = np.asarray(points)
        pts_3D, pts_2D = filter_lidar_points(lidar2camera, point_cloud, (frame.shape[1], frame.shape[0]))

        # For each object detected by the YOLOv8 model, fuse and process it
        all_corners_3D = []
        all_filtered_points_of_object = []
        all_object_IDs = []
        objects3d_data = []
        for j, cls in enumerate(boxes.cls.tolist()):
            conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
            box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else None

            all_object_IDs.append(box_id)

            # Check if the mask is empty before processing
            if masks.xy[j].size == 0:
                continue

            # Pass the segmentation mask to the fusion function
            fusion_result = lidar_camera_fusion(pts_3D, pts_2D, frame, masks.xy[j], int(cls), lidar2camera, erosion_factor=erosion_factor, depth_factor=depth_factor, PCA=self.pca)

            # If the fusion is successfull, retrieve relevant bbox data (e.g. for RoboCar)
            if fusion_result is not None:
                filtered_points_of_object, corners_3D, yaw = fusion_result

                all_corners_3D.append(corners_3D)
                all_filtered_points_of_object.append(filtered_points_of_object)

                # Retrieve the ROS data (e.g. relevant for RoboCar)
                ROS_type = int(np.int32(cls))
                bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                ROS_ground_center = np.mean(corners_3D[bottom_indices], axis=0)
                ROS_dimensions = np.ptp(corners_3D, axis=0)                
                ROS_points = corners_3D
                time_between_frames = 0.1

                # Compute the velocity and direction (only available with tracking)
                if box_id in self.last_ground_center_of_id and not np.array_equal(self.last_ground_center_of_id[box_id], ROS_ground_center):
                    ROS_direction, ROS_velocity = compute_relative_object_velocity(self.last_ground_center_of_id[box_id], ROS_ground_center, time_between_frames)
                else:
                    ROS_direction = None
                    ROS_velocity = None

                self.last_ground_center_of_id[box_id] = ROS_ground_center

                # Save the ROS information of the current object and append it to an array that contains all information of all objects in the frame
                if ROS_type is not None and ROS_ground_center is not None and ROS_direction is not None and ROS_dimensions is not None and ROS_velocity is not None and ROS_points is not None:
                    objects3d_data.append([ROS_type, ROS_ground_center, ROS_direction, ROS_dimensions, ROS_velocity, ROS_points])


        return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
    
    def get_IoU_results(self, frame, pts, lidar2camera, erosion_factor, depth_factor):
        if self.tracking:
            results = self.model.track(
                source=frame,
                classes=[0, 1, 2, 3, 5, 6, 7],
                verbose=False,
                show=False,
                persist=True,
                tracker='bytetrack.yaml'
            )
        else:  
            results = self.model.predict(
                source=frame,
                classes=[0, 1, 2, 3, 5, 6, 7],
                verbose=False,
                show=False,
            )

        # Get the results from the YOLOv8-seg model
        r = results[0]
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs

        # Preprocess LiDAR point cloud
        points = np.fromfile(pts, dtype=np.float32).reshape((-1, 4))[:, 0:3]
        point_cloud = np.asarray(points)
        pts_3D, pts_2D = filter_lidar_points(lidar2camera, point_cloud, (frame.shape[1], frame.shape[0]))

        # For each object detected by the YOLOv8 model, fuse and process it
        all_corners_3D = []
        all_filtered_points_of_object = []
        objects3d_data = []
        for j, cls in enumerate(boxes.cls.tolist()):
            conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
            box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else None

            # Check if the mask is empty before processing
            if masks.xy[j].size == 0:
                continue

            # Pass the segmentation mask to the fusion function
            fusion_result = lidar_camera_fusion(pts_3D, pts_2D, frame, masks.xy[j], int(cls), lidar2camera, erosion_factor=erosion_factor, depth_factor=depth_factor, PCA=self.pca)

            # If the fusion is successfull, retrieve the relevant data for the IoU computation with KITTI GT boxes
            if fusion_result is not None:
                filtered_points_of_object, corners_3D, yaw = fusion_result

                all_corners_3D.append(corners_3D)
                all_filtered_points_of_object.append(filtered_points_of_object)

                if cls == 0:
                    type = "Pedestrian"
                elif cls == 1:
                    type = "Cyclist"
                elif cls == 2:
                    type = "Car"
                else:
                    type = "DontCare"

                # Ground Center is the center of the bottom bbox side, thus of the 4 corners with the lowest z value (in LiDAR coordinates) 
                bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                ground_center = np.mean(corners_3D[bottom_indices], axis=0)

                # Get the bbox dimensions in l, w, h format
                dimensions = np.ptp(corners_3D, axis=0)

                # Append relevant information to array that is later returned
                objects3d_data.append([type, ground_center, dimensions, yaw])

        return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object