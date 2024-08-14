import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_relative_object_velocity(ground_center_frame1, ground_center_frame2, time_between_frames):
    # Compute displacement vector between ground centers
    displacement = ground_center_frame2 - ground_center_frame1

    # Compute relative velocity components
    relative_velocity_x = displacement[0] / time_between_frames
    relative_velocity_y = displacement[1] / time_between_frames
    relative_velocity_z = displacement[2] / time_between_frames

    # Return velocity vector
    return displacement, np.array([relative_velocity_x, relative_velocity_y, relative_velocity_z])


def compute_box_3d(dimensions, location, rotation_y):
    """
    Compute the 8 corners of a 3D bounding box.
    
    Args:
    - dimensions (tuple): (height, width, length)
    - location (tuple): (x, y, z)
    - rotation_y (float): Rotation around the Y-axis in radians
    
    Returns:
    - corners_3d (np.ndarray): 8x3 array of the 3D corners of the bounding box
    """
    h, w, l = dimensions
    x, y, z = location
    
    # 3D bounding box corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    # Rotation matrix around the Y-axis
    R = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])
    
    # Apply rotation and then translation
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    return corners_3d.T

def get_gt_bbox_edges(corners_2D):
    edges = []
    
    # Define the edges based on the common bounding box structure
    edges_indices = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Top edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Bottom edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    
    # Map the corner indices to actual points
    for (start_idx, end_idx) in edges_indices:
        edges.append((corners_2D[start_idx], corners_2D[end_idx]))

    return edges

def get_pred_bbox_edges(corners_2D):
    edges = []
    
    # Define the edges in the desired order
    edges_indices = [
        (1, 7), (1, 6), (4, 7), (4, 6), # Top edges
        (0, 2), (0, 3), (2, 5), (3, 5), # Bottom eges
        (0, 1), (2, 7), (3, 6), (4, 5), # Vertical edges
    ]
    
    # Map the corner indices to actual points
    for (start_idx, end_idx) in edges_indices:
        edges.append((corners_2D[start_idx], corners_2D[end_idx]))

    return edges


def get_axis_aligned_bbox(center, size, rotation_y):
    """ Convert a rotated 3D bounding box to an axis-aligned bounding box (AABB). """
    h, w, l = size
    x, y, z = center

    # Define the 8 corners of the bounding box
    corners = np.array([
        [l/2, 0, w/2],
        [l/2, 0, -w/2],
        [-l/2, 0, -w/2],
        [-l/2, 0, w/2],
        [l/2, -h, w/2],
        [l/2, -h, -w/2],
        [-l/2, -h, -w/2],
        [-l/2, -h, w/2]
    ])

    # Apply the rotation matrix
    rotation_matrix = R.from_euler('y', rotation_y).as_matrix()
    corners = np.dot(corners, rotation_matrix.T)

    # Translate corners
    corners += center

    # Compute axis-aligned bounding box
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)

    aabb_center = (min_corner + max_corner) / 2
    aabb_size = max_corner - min_corner

    return aabb_center, aabb_size
