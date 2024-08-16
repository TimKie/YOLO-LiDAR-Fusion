import numpy as np
import os
import time
import cv2
from utils import *
from calibration import *
from visualization import *

def calculate_3d_iou(center1, size1, rotation_y1, center2, size2, rotation_y2):
    """ Calculate IoU between two 3D bounding boxes with yaw. """
    # Convert rotated bounding boxes to axis-aligned bounding boxes
    aabb_center1, aabb_size1 = get_axis_aligned_bbox(center1, size1, rotation_y1)
    aabb_center2, aabb_size2 = get_axis_aligned_bbox(center2, size2, rotation_y2)

    # Compute IoU for axis-aligned bounding boxes
    min1 = aabb_center1 - aabb_size1 / 2
    max1 = aabb_center1 + aabb_size1 / 2
    min2 = aabb_center2 - aabb_size2 / 2
    max2 = aabb_center2 + aabb_size2 / 2
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)

    intersection_size = np.maximum(intersection_max - intersection_min, 0)
    intersection_volume = np.prod(intersection_size)

    volume1 = np.prod(aabb_size1)
    volume2 = np.prod(aabb_size2)

    union_volume = volume1 + volume2 - intersection_volume

    if union_volume == 0:
        return 0.0  # To avoid division by zero

    iou = intersection_volume / union_volume

    return iou


# Function to match pred bboxes with gt bboxes (best IoU values demonstrate that corresponding bboxes are found)
def find_best_ious(gt_centers, gt_sizes, gt_rotations, det_centers, det_sizes, det_rotations):
    num_gt_bboxes = len(gt_centers)
    num_det_bboxes = len(det_centers)
    best_ious = np.zeros(num_gt_bboxes)
    best_det_indices = np.full(num_gt_bboxes, -1, dtype=int)  # Initialize with -1 for cases with no matches

    for gt_index in range(num_gt_bboxes):
        best_iou = 0
        best_det_index = -1
        for det_index in range(num_det_bboxes):
            center1 = np.array(gt_centers[gt_index])
            size1 = np.array(gt_sizes[gt_index])
            rotation_y1 = np.array(gt_rotations[gt_index])
            center2 = np.array(det_centers[det_index])
            size2 = np.array(det_sizes[det_index])
            rotation_y2 = np.array(det_rotations[det_index])
            iou = calculate_3d_iou(center1, size1, rotation_y1, center2, size2, rotation_y2)
            if iou > best_iou:
                best_iou = iou
                best_det_index = det_index
        best_ious[gt_index] = best_iou
        best_det_indices[gt_index] = best_det_index

    return best_ious, best_det_indices


def evaluate_single_data(dataset_path, output_path, image_index, detector, erosion, depth, categories=['Car', 'Pedestrian', 'Cyclist']):
    # Set initial IoU values
    max_iou = -1
    avg_iou = -1

    # Read the files specified by the argument
    image_path = os.path.join(dataset_path, f"data_object_image_2/training/image_2/{image_index}.png")
    velodyne_path = os.path.join(dataset_path, f"data_object_velodyne/training/velodyne/{image_index}.bin")
    calib_path = os.path.join(dataset_path, f"data_object_calib/training/calib/{image_index}.txt")
    label_path = os.path.join(dataset_path, f"data_object_label_2/training/label_2/{image_index}.txt")

    # Initialize dictionaries to store GT bbox information for each object category
    gt_centers = {cat: [] for cat in categories}
    gt_sizes = {cat: [] for cat in categories}
    gt_rotations = {cat: [] for cat in categories}
    
    # Initilaize arrays to store labels and colors for all bboxes (used for visualization)
    labels = []
    colors = []

    # Gather GT bbox information
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.split()
            if parts[0] in gt_centers:
                category = parts[0]
                d = [float(parts[8]), float(parts[9]), float(parts[10])]
                c = [float(parts[11]), float(parts[12]), float(parts[13])]
                gt_centers[category].append(c)
                gt_sizes[category].append(d)
                labels.append(f"gt_box")
                colors.append("blue")
                gt_rotations[category].append(float(parts[14]))
    
    # Read the image
    frame = cv2.imread(image_path)
    
    # Initialize LiDAR to Camera calibration object
    lidar2cam = LiDAR2Camera(calib_path)

    # Fuse and process the image and corresponding point cloud
    objects3d_data, all_pred_corners_3D, point_cloud_3D, point_cloud_2D, all_filtered_points_of_object = detector.get_IoU_results(frame, velodyne_path, lidar2cam, erosion_factor=erosion, depth_factor=depth)

    # Initialize dictionnaries to store prediction bbox information for each object category
    predicted_centers = {cat: [] for cat in categories}
    predicted_sizes = {cat: [] for cat in categories}
    pred_rotations = {cat: [] for cat in categories}

    # Save prediction bbox information
    for pred_obj in objects3d_data:
        obj_type, ground_center, dimensions, yaw = pred_obj
        if obj_type in predicted_centers:
            new_dimensions = [dimensions[2], dimensions[0], dimensions[1]]
            camera_3D_ground_center = lidar2cam.convert_single_3D_to_camera_coords(np.array(ground_center))
            predicted_centers[obj_type].append(camera_3D_ground_center.tolist())
            predicted_sizes[obj_type].append(new_dimensions)
            labels.append(f"pred_box")
            colors.append("red")
            pred_rotations[obj_type].append(yaw)

    # Initialize dictionary to store IoU scores for each category
    category_ious = {cat: [] for cat in categories}

    # Sort bbox information based on categories for later IoU computation
    for category in gt_centers.keys():
        gt_centers_cat = gt_centers[category]
        gt_sizes_cat = gt_sizes[category]
        gt_rotations_cat = gt_rotations[category]
        pred_centers_cat = predicted_centers[category]
        pred_sizes_cat = predicted_sizes[category]
        pred_rotations_cat = pred_rotations[category]

        # Compute the best matching IoU scores for each bbox
        best_ious, best_det_indices = find_best_ious(gt_centers_cat, gt_sizes_cat, gt_rotations_cat, pred_centers_cat, pred_sizes_cat, pred_rotations_cat)

        # Set best IoU scores when new ones are available
        if len(best_ious) != 0:
            max_iou = max(best_ious)
            avg_iou = np.mean(best_ious)
            category_ious[category].append(max_iou)
            category_ious[category].append(avg_iou)

    # Print the best and average IoU scores for each category that is available on the specified image
    for category, ious in category_ious.items():
        if ious:
            print(f"\n-------- {category} --------")
            print(f"Best IoU: {ious[0]}")
            print(f"Average IoU: {ious[1]}")
    print()

    # ---- Gather inforamtion to visualize the GT and prediction bboxes in 3D, projected onto the image and in BEV ----
    all_centers = []
    all_gt_centers = []
    all_predicted_centers = []
    for key in gt_centers:
        all_centers.extend(gt_centers[key])
        all_gt_centers.extend(gt_centers[key])
    for key in predicted_centers:
        all_centers.extend(predicted_centers[key])
        all_predicted_centers.extend(predicted_centers[key])

    all_sizes = []
    all_gt_sizes = []
    all_predicted_sizes = []
    for key in gt_sizes:
        all_sizes.extend(gt_sizes[key])
        all_gt_sizes.extend(gt_sizes[key])
    for key in predicted_sizes:
        all_sizes.extend(predicted_sizes[key])
        all_predicted_sizes.extend(predicted_sizes[key])

    all_rotations = []
    all_gt_rotations = []
    all_pred_rotations = []
    for key in gt_rotations:
        all_rotations.extend(gt_rotations[key])
        all_gt_rotations.extend(gt_rotations[key])
    for key in pred_rotations:
        all_rotations.extend(pred_rotations[key])
        all_pred_rotations.extend(pred_rotations[key])

    #plot_3d_bounding_boxes(all_centers, all_sizes, all_rotations, labels, colors)

    # Plot the GT bounding boxes
    for i, center in enumerate(all_gt_centers):
        gt_corners_3D = compute_box_3d(all_gt_sizes[i], center, all_gt_rotations[i])
        plot_projected_gt_bounding_boxes(lidar2cam, frame, gt_corners_3D, (255, 0, 0))

    for pred_corner_3D in all_pred_corners_3D:
        plot_projected_pred_bounding_boxes(lidar2cam, frame, pred_corner_3D, (0, 0, 255))

    draw_projected_3D_points(lidar2cam, frame, point_cloud_3D, point_cloud_2D, np.vstack(all_filtered_points_of_object))

    # Create a bev representation
    bev = create_BEV(all_filtered_points_of_object, all_pred_corners_3D)
    

    if output_path == "show":
        # Show the combined image
        create_combined_image(frame, bev, output_path)
    else:
         # Save the combined image for visualization
        create_combined_image(frame, bev, output_path)
        print(f"Processed Image saved in {output_path}")



def evaluate_dataset(dataset_path, output_path, detector, erosion, depth, categories=['Car', 'Pedestrian', 'Cyclist']):
    best_iou = -1
    best_erosion = None
    best_depth = None
    best_avg_iou = -1
    best_avg_erosion = None
    best_avg_depth = None
    max_iou = -1
    avg_iou = -1
    image_index_max_iou = -1
    image_index_avg_iou = -1

    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping computation.")
        return
    
    print(f"Results will be stored in {output_path}")
    os.makedirs(output_path)

    # Read the files specified by the argument
    base_image_path = os.path.join(dataset_path, f"data_object_image_2/training/image_2/")
    base_velodyne_path = os.path.join(dataset_path, f"data_object_velodyne/training/velodyne/")
    base_calib_path = os.path.join(dataset_path, f"data_object_calib/training/calib/")
    base_label_path = os.path.join(dataset_path, f"data_object_label_2/training/label_2/")

    for filename in sorted(os.listdir(base_image_path)):
        index = filename[0:-4]
        
        image_path = base_image_path + index + ".png"
        velodyne_path = base_velodyne_path + index + ".bin"
        calib_path = base_calib_path + index + ".txt"
        label_path = base_label_path + index + ".txt"

        # Initialize dictionaries to store GT bbox information for each object category
        gt_centers = {cat: [] for cat in categories}
        gt_sizes = {cat: [] for cat in categories}
        gt_rotations = {cat: [] for cat in categories}
        
        # Initilaize arrays to store labels and colors for all bboxes (used for visualization)
        labels = []
        colors = []

        # Gather GT bbox information
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.split()
                if parts[0] in gt_centers:
                    category = parts[0]
                    d = [float(parts[8]), float(parts[9]), float(parts[10])]
                    c = [float(parts[11]), float(parts[12]), float(parts[13])]
                    gt_centers[category].append(c)
                    gt_sizes[category].append(d)
                    labels.append(f"gt_box")
                    colors.append("blue")
                    gt_rotations[category].append(float(parts[14]))

        start_time = time.time()

        # Read the image
        frame = cv2.imread(image_path)
        
        # Initialize LiDAR to Camera calibration object
        lidar2cam = LiDAR2Camera(calib_path)

        # Fuse and process the image and corresponding point cloud
        objects3d_data, _, _, _, _ = detector.get_IoU_results(frame, velodyne_path, lidar2cam, erosion_factor=erosion, depth_factor=depth)

        # Compute the processing time for the current frame
        processing_time = time.time() - start_time

        # Initialize dictionnaries to store prediction bbox information for each object category
        predicted_centers = {cat: [] for cat in categories}
        predicted_sizes = {cat: [] for cat in categories}
        pred_rotations = {cat: [] for cat in categories}

        # Save prediction bbox information
        for pred_obj in objects3d_data:
            obj_type, ground_center, dimensions, yaw = pred_obj
            if obj_type in predicted_centers:
                new_dimensions = [dimensions[2], dimensions[0], dimensions[1]]
                camera_3D_ground_center = lidar2cam.convert_single_3D_to_camera_coords(np.array(ground_center))
                predicted_centers[obj_type].append(camera_3D_ground_center.tolist())
                predicted_sizes[obj_type].append(new_dimensions)
                labels.append(f"pred_box")
                colors.append("red")
                pred_rotations[obj_type].append(yaw)

        # Initialize dictionary to store IoU scores for each category
        category_ious = {cat: [] for cat in categories}

        # Sort bbox information based on categories for later IoU computation
        for category in gt_centers.keys():
            gt_centers_cat = gt_centers[category]
            gt_sizes_cat = gt_sizes[category]
            gt_rotations_cat = gt_rotations[category]
            pred_centers_cat = predicted_centers[category]
            pred_sizes_cat = predicted_sizes[category]
            pred_rotations_cat = pred_rotations[category]

            # Compute the best matching IoU scores for each bbox
            best_ious, best_det_indices = find_best_ious(gt_centers_cat, gt_sizes_cat, gt_rotations_cat, pred_centers_cat, pred_sizes_cat, pred_rotations_cat)

            # Set best IoU scores when new ones are available
            if len(best_ious) != 0:
                max_iou = max(best_ious)
                avg_iou = np.mean(best_ious)
                category_ious[category].append((max_iou, avg_iou))

        # Save the result of each frame in a text file
        txt_file_path = os.path.join(output_path, f"{index}.txt")
        with open(txt_file_path, "w") as file:
            file.write(f"erosion={erosion} depth={depth}\n")
            file.write(f"Processing Time: {processing_time}\n")
            for category, ious in category_ious.items():
                if ious:
                    max_iou = max([iou[0] for iou in ious])
                    avg_iou = np.mean([iou[1] for iou in ious])
                    file.write(f"{category} best_IoU={max_iou:.4f} average_IoU={avg_iou:.4f}\n")

                    if max_iou > best_iou:
                        best_iou = max_iou
                        best_erosion = erosion
                        best_depth = depth
                        image_index_max_iou = index

                    if avg_iou > best_avg_iou:
                        best_avg_iou = avg_iou
                        best_avg_erosion = erosion
                        best_avg_depth = depth
                        image_index_avg_iou = index

    # Save the overall best result in a text file
    txt_file_path_best_para = os.path.join(output_path, "overall_best_parameters.txt")
    with open(txt_file_path_best_para, "w") as file:
        file.write(f"Best IoU: erosion={best_erosion} depth={best_depth} best_IoU={best_iou:.4f} image_index_max_iou={image_index_max_iou}\n")
        file.write(f"Best Average IoU: erosion={best_avg_erosion} depth={best_avg_depth} average_IoU={best_avg_iou:.4f} image_index_avg_iou={image_index_avg_iou}\n")
