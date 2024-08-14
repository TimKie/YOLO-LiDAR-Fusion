from calibration import *
from data_processing import *


def lidar_camera_fusion(pts_3D, pts_2D, frame, seg_mask, obj_class, lidar2cam, erosion_factor, depth_factor, PCA):
    """ Fuse 3D LiDAR Points with Camera frame and return filtered 3D LiDAR Points and 3D Bounding Box of current object """

    #print("\nFuse LiDAR points of detected objects with image ...")

    # Erode the segmentation polygon
    eroded_seg = erode_polygon(seg_mask, frame, erosion_factor)

    if eroded_seg is None or len(eroded_seg) <= 2:
        #print("\n\tPass object (segmentation is too small)")
        return None
    
    all_points_of_object = []

    """ ----------------- FUSION -----------------"""
    # Create a mask for the eroded polygon
    mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [eroded_seg.astype(np.int32)], color=1)

    # Convert 2D points to integer coordinates
    pts_2D_int = pts_2D.astype(np.int32)
    
    # Filter points using the mask
    inside_mask_indices = mask[pts_2D_int[:, 1], pts_2D_int[:, 0]] == 1

    # Select corresponding 3D points
    all_points_of_object = pts_3D[inside_mask_indices]
    """ ------------------------------------------"""

    # Skip objects with no points
    if len(all_points_of_object) == 0:
        return None

    # Filter points based on their distances
    filtered_points_of_object = filter_points_with_depth_clustering(np.array(all_points_of_object), depth_factor=depth_factor)

    # Skip objects with insufficient points and if points are coplanar (e.g. all points have the same z coordinate)
    if len(filtered_points_of_object) < 4 or is_coplanar(filtered_points_of_object):
        #print("\n\tPass object (not enough points after filtering)")
        return None

    # Get the edges and corners of the 3D bounding box
    yaw = 0
    if PCA:
        # Compute the oriented 3D bounding box (discard the rotation along the z-axis for detected objects other than persons)
        if obj_class == 0:
            bbox_corners_3D, yaw = create_bbox_3D_PCA(filtered_points_of_object, lidar2cam)
        else:
            # Use PCA without z-axis rotation for other classes
            bbox_corners_3D, yaw = create_bbox_3D_PCA_no_z_rotation(filtered_points_of_object, lidar2cam)
    
    # Note: Pedestrian class always uses PCA 
    else:
        # Compute the oriented 3D bounding box (discard the rotation along the z-axis for detected objects other than persons)
        if obj_class == 0:
            bbox_corners_3D, yaw = create_bbox_3D_PCA(filtered_points_of_object, lidar2cam)
        else:
            # Don't use PCA for classes other than Pedestrians
            bbox_corners_3D = create_bbox_3D(filtered_points_of_object, lidar2cam)

    return filtered_points_of_object, bbox_corners_3D, yaw