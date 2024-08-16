import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import open3d as o3d


def filter_lidar_points(lidar2cam, pts_3D, size, print_info=False):
    """ Filter 3D LiDAR Points to keep only those which lie inside the Image FOV """

    if print_info:
        print("\nRemove 3D LiDAR Points outside the Image FOV ...")

    # Initialize the image boundaries
    xmin = 0
    ymin = 0
    xmax = size[0]
    ymax = size[1]

    # Define the clip distance of the sensor
    clip_distance = 1

    # Convert the 3D LiDAR points to 2D image corrdinates
    all_pts_2D = lidar2cam.convert_3D_to_2D(pts_3D)

    # Create a boolean mask that checks if a 3D LiDAR point lies within the image boundaries 
    inside_pts_indices = ((all_pts_2D[:, 0] >= xmin) & (all_pts_2D[:, 0] < xmax) & (all_pts_2D[:, 1] >= ymin) & (all_pts_2D[:, 1] < ymax))

    # Add constraint to boolean mask to only keep points that are clip_distance away from the sensor
    inside_pts_indices = inside_pts_indices & (pts_3D[:, 0] > clip_distance)

    # Apply boolean mask to receive only 3D and 2D points that lie inside the image
    pts_3D_inside_img = pts_3D[inside_pts_indices, :]
    pts_2D_inside_img = all_pts_2D[inside_pts_indices, :]

    return pts_3D_inside_img, pts_2D_inside_img
    

def is_coplanar(points, print_info=False):
    """ Check if an array of 3D LiDAR Points is coplanar """

    if print_info:
        print("\nCheck if 3D LiDAR Points are coplanar ...")

    # Only check arrays of more than 4 points 
    if len(points) < 4:
        return True
    
    # Compute the covariance matrix and its eigenvalues
    cov_matrix = np.cov(points.T)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    
    # If one of the eigenvalues is close to zero, points are nearly coplanar
    return np.min(eigenvalues) < 1e-6


def erode_polygon(polygon, img, erosion_factor=25, print_info=False):
    """ Return an Eroded Polygon based on an Erosion Factor """

    if print_info:
        print(f"\nErode Segmentation Mask with erosion_factor = {erosion_factor} ...")

    # Check if the polygon is empty
    if len(polygon) == 0:
        return None
    
    # Calculate the area of the polygon
    polygon_area = cv2.contourArea(polygon.astype(np.int32))

    # Calculate the scaled erosion factor based on the square root of the polygon area
    scaled_erosion_factor = np.sqrt(polygon_area) / erosion_factor

    # Create a mask for the polygon
    mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], color=255)

    # Define the erosion kernel size based on the erosion factor
    kernel_size = int(2 * scaled_erosion_factor + 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion to the mask
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Find contours of the eroded mask
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours were found (polygon is too small)
    if len(contours) == 0:
        return None

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Convert the contour to the polygon format
    eroded_polygon = np.squeeze(largest_contour).astype(np.float32)

    return eroded_polygon


def filter_points_with_depth_clustering(points_of_object, eps=0.5, depth_factor=20, print_info=False):
    """ Return filtered 3D LiDAR Points using DBSCAN """

    if print_info:
        print("\nFilter 3D LiDAR Points using DBSCAN ...")

    # Compute min_samples as a percentage of number of points
    min_samples = max(5, int(len(points_of_object) * 0.01))
    
    # Compute the depth values using the Euclidean Distance
    depths = np.sqrt(points_of_object[:, 0]**2 + points_of_object[:, 1]**2)

    # Density-Based Clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(depths.reshape(-1, 1))

    # Calculate the number of points in each cluster
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    cluster_counts_dict = dict(zip(unique_clusters, cluster_counts))

    # Sort clusters based on the number of points they contain
    sorted_clusters = sorted(cluster_counts_dict, key=cluster_counts_dict.get, reverse=True)

    # Keep only the clusters with the most points
    top_cluster = sorted_clusters[:1]

    # Dynamic Depth Range
    depth_ranges = {}
    for cluster_label in top_cluster:
        if cluster_label == -1:
            continue  # Skip noise points
        cluster_depth_values = depths[clusters == cluster_label]
        min_depth = np.min(cluster_depth_values)
        max_depth = np.max(cluster_depth_values)
        # Adjust the depth range by a factor
        range_length = max_depth - min_depth
        min_depth_adjusted = min_depth + (1 - depth_factor) * range_length / 2
        max_depth_adjusted = max_depth - (1 - depth_factor) * range_length / 2
        depth_ranges[cluster_label] = (min_depth_adjusted, max_depth_adjusted)

    # Filtering
    filtered_points_of_object = []
    for cluster_label, depth_range in depth_ranges.items():
        min_depth, max_depth = depth_range
        valid_cluster_mask = clusters == cluster_label
        valid_cluster_indices = np.where(valid_cluster_mask)[0]
        cluster_points = points_of_object[valid_cluster_indices]
        depth_mask = (cluster_points[:, 0] >= min_depth) & (cluster_points[:, 0] <= max_depth)
        filtered_points_of_object.extend(cluster_points[depth_mask])

    return np.array(filtered_points_of_object)


""" ------------------------------------------ 3D BOUNDING BOXES FUNCTIONS ------------------------------------------ """
def create_bbox_3D(pts_3D, lidar2cam, print_info=False):
    """ Create simple 3D Bounding Box """

    if print_info:
        print("\nCreate simple 3D Bounding Box ...")

    # Determine bounding box corners
    min_point = np.min(pts_3D, axis=0)
    max_point = np.max(pts_3D, axis=0)

    # Define the corners of the bounding box
    corners_3D = np.array([
        [min_point[0], min_point[1], min_point[2]],  # Corner 0
        [min_point[0], min_point[1], max_point[2]],  # Corner 1
        [min_point[0], max_point[1], min_point[2]],  # Corner 2
        [max_point[0], min_point[1], min_point[2]],  # Corner 3
        [max_point[0], max_point[1], max_point[2]],  # Corner 4
        [max_point[0], max_point[1], min_point[2]],  # Corner 5
        [max_point[0], min_point[1], max_point[2]],  # Corner 6
        [min_point[0], max_point[1], max_point[2]]   # Corner 7
    ])

    return corners_3D


def create_bbox_3D_PCA(pts_3D, lidar2cam, print_info=False):
    """ Create 3D Bounding Box using PCA """

    if print_info:
        print("\nCreate 3D Bounding Box using PCA ...")

    # Convert points from LiDAR to camera coordinates
    pts_3D_camera_coord = lidar2cam.convert_3D_to_camera_coords(pts_3D)

    # Create a point cloud with the camera coordinate points
    point_cloud_2 = o3d.geometry.PointCloud()
    point_cloud_2.points = o3d.utility.Vector3dVector(pts_3D_camera_coord)

    # Compute the oriented bounding box in camera coordinates
    obb2 = point_cloud_2.get_oriented_bounding_box()

    # Get the rotation matrix in camera coordinates
    R_camera_coord = np.array(obb2.R)

    # Calculate the yaw (rotation around the y-axis in camera coordinates)
    yaw = np.arctan2(R_camera_coord[0, 0], R_camera_coord[2, 0])

    # Create a point cloud from the points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts_3D)

    # Compute the oriented bounding box
    obb = point_cloud.get_oriented_bounding_box()

    # Get the 3D corners of the bounding box
    corners_3D = np.asarray(obb.get_box_points())

    return corners_3D, yaw


def create_bbox_3D_PCA_no_z_rotation(pts_3D, lidar2cam, print_info=False):
    """ Create 3D Bounding Box using PCA without z-axis rotation """

    if print_info:
        print("\nCreate 3D Bounding Box using PCA without z-axis rotation ...")
        
    # Convert points from LiDAR to camera coordinates
    pts_3D_camera_coord = lidar2cam.convert_3D_to_camera_coords(pts_3D)

    # Create a point cloud with the camera coordinate points
    point_cloud_2 = o3d.geometry.PointCloud()
    point_cloud_2.points = o3d.utility.Vector3dVector(pts_3D_camera_coord)

    # Compute the oriented bounding box in camera coordinates
    obb2 = point_cloud_2.get_oriented_bounding_box()

    # Get the rotation matrix in camera coordinates
    R_camera_coord = np.array(obb2.R)

    # Calculate the yaw (rotation around the y-axis in camera coordinates)
    yaw = np.arctan2(R_camera_coord[0, 0], R_camera_coord[2, 0])

    # Create a point cloud from the points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts_3D)

    # Compute the oriented bounding box using PCA
    obb = point_cloud.get_oriented_bounding_box()

    # Get the rotation matrix of the oriented bounding box
    R = np.array(obb.R)

    # Change the rotation matrix
    R[:, 0] = [R[0, 0], R[1, 0], 0]     # Ensure x-axis rotation is only in x-y plane
    R[:, 1] = [R[0, 1], R[1, 1], 0]     # Ensure y-axis rotation is only in x-y plane
    R[:, 2] = [0, 0, 1]                 # Ensure z-axis rotation is discarded

    # No rotation along x-axis
    # R[:, 0] = [1, 0, 0]
    # R[:, 1] = [0, R[1, 1], R[2, 1]]
    # R[:, 2] = [0, R[1, 2], R[2, 2]]

    # No rotation along y-axis
    # R[:, 0] = [R[0, 0], 0, R[2, 0]]
    # R[:, 1] = [0, 1, 0]
    # R[:, 2] = [R[0, 2], 0, R[2, 2]]

    # Create a new oriented bounding box with the adjusted rotation matrix
    obb_new = o3d.geometry.OrientedBoundingBox(obb.center, R, obb.extent)

    # Get the 3D corners of the bounding box
    corners_3D = np.asarray(obb_new.get_box_points())

    return corners_3D, yaw
