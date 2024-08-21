import argparse
import os
from random import randint
from calibration import *
from data_processing import *
from fusion import *
from detector import *
from utils import *
from evaluation import *
from visualization import *


if __name__ == "__main__":
    # Define the base dataset path
    KITTI_dataset_path = "../../KITTI_dataset/"

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add a mandatory argument for the image index
    parser.add_argument('image_index', type=str, help="number of the image (between '000000' and '007517') or 'random' for multiple random images or 'evaluation' to process the complete dataset or 'video' to process raw data and create a video")

    # Add optional arguments for detection or tracking, model size, dataset path, erosion, depth and output path
    parser.add_argument('--mode', dest='detection_or_tracking', type=str, default="detect", choices=['detect', 'track'], help="specify if the model should use detection only ('detect') or also tracking ('track')")
    parser.add_argument('--model-size', dest='model_size', type=str, default="m", choices=['n', 's', 'm', 'l', 'x'], help="specify the size of the YOLOv8 model (n, s, m, l, x)")
    parser.add_argument('--erosion', dest='erosion', type=int, default=25, help="specify the erosion value to be used by the model")
    parser.add_argument('--depth', dest='depth', type=int, default=20, help="specify the depth value to be used by the model")
    parser.add_argument('--pca', dest='pca', type=bool, default=False, help="specify wheter PCA should be used to create the 3D bounding boxes for all detected objects")
    parser.add_argument('--dataset-path', dest='dataset_path', type=str, default="../KITTI_dataset/", help="specify the relative path to the KITTI dataset")
    parser.add_argument('--output-path', dest='output_path', type=str, default="", help="specify the relative path where the output should be saved")
    parser.add_argument('--image-amount', dest='image_amount', type=int, default=10, help="specify the desired amount of random images (only when first argument is 'random')")
    parser.add_argument('--video-dir', dest='video_directory', type=str, default="../KITTI_raw_data", help="specify the relative path of the directory that contains the ordered frames of the video that the model should process (only when first argument is 'video')")

    # Parse the arguments to include the conditionally added arguments
    args = parser.parse_args()

    # Define the desired model
    model = f"yolov8{args.model_size}-seg.pt"

    # Define the detector
    if args.detection_or_tracking == "detect" and args.pca:
        detector = YOLOv8Detector(model, tracking=False, PCA=True)
    elif args.detection_or_tracking == "detect" and not args.pca:
        detector = YOLOv8Detector(model, tracking=False, PCA=False)
    elif args.detection_or_tracking == "track" and args.pca:
        detector = YOLOv8Detector(model, tracking=True, PCA=True)
    elif args.detection_or_tracking == "track" and not args.pca:
        detector = YOLOv8Detector(model, tracking=True, PCA=False)

    # Apply model to data specified as arguments
    if args.image_index.isdigit() and len(args.image_index) == 6 and 0 <= int(args.image_index) <= 7517:
        if len(args.output_path) != 0:
            os.makedirs(args.output_path, exist_ok=True)
            output_path = os.path.join(args.output_path, f"processed_image_{args.image_index}.png")
        else:
            output_path = "show"
        evaluate_single_data(KITTI_dataset_path, output_path, args.image_index, detector, args.erosion, args.depth)

    elif args.image_index == "evaluation":
        print(f"\nProcessing complete dataset at location {KITTI_dataset_path}")
        print(f"Model = {model} \tMode = {args.detection_or_tracking} \terosion_factor = {args.erosion} \tdepth_factor = {args.depth}")

        evaluate_dataset(KITTI_dataset_path, args.output_path, detector, args.erosion, args.depth)

    elif args.image_index == "video":
        print(f"\nProcessing Frames in Input Directory {args.video_directory} ...")

        # Read the files specified by the argument
        base_image_path = os.path.join(args.video_directory, f"image_02/data/")
        base_velodyne_path = os.path.join(args.video_directory, f"velodyne_points/data/")
        base_calib_path = os.path.join(args.video_directory, f"calib/")

        c2c_calib_path = os.path.join(base_calib_path, "calib_cam_to_cam.txt")
        v2c_calib_path = os.path.join(base_calib_path, "calib_velo_to_cam.txt")

        if len(args.output_path) == 0:
            output_path_video = "../Model_Output/Results_Video/"
        else:
            output_path_video = os.path.join(args.output_path, "Results_Video/")
        
        os.makedirs(output_path_video, exist_ok=True)

        # Filter and sort files in the output video directory, ignoring hidden files
        video_files = sorted([f for f in os.listdir(output_path_video) if not f.startswith('.')])

        # Get the last video index if there are any files
        if video_files:
            last_video_index = int(video_files[-1][-5])
        else:
            last_video_index = 0

        # Initilaize array to store processed images
        all_processed_frames = []
        all_bev = []

        # Filter and sort files in the base image path, ignoring hidden files
        image_files = sorted([f for f in os.listdir(base_image_path) if not f.startswith('.')])
        for filename in image_files:
            index = filename[0:-4]
            
            # Define the paths for the image, LiDAR point cloud and calibration files
            image_path = base_image_path + index + ".png"
            velodyne_path = base_velodyne_path + index + ".bin"

            # Read the image
            frame = cv2.imread(image_path)

            # Initialize LiDAR to Camera calibration object
            lidar2cam = LiDAR2Camera_KITTI_raw_data(c2c_calib_path, v2c_calib_path)

            # Fuse and process the image and corresponding point cloud
            objects3d_data, all_pred_corners_3D, point_cloud_3D, point_cloud_2D, all_filtered_points_of_object, all_object_IDs = detector.process_frame(frame, velodyne_path, lidar2cam, erosion_factor=args.erosion, depth_factor=args.depth)

            for i, pred_corner_3D in enumerate(all_pred_corners_3D):
                plot_projected_pred_bounding_boxes(lidar2cam, frame, pred_corner_3D, (0, 0, 255), all_object_IDs[i])

            all_filtered_points_of_object_combined = np.vstack(all_filtered_points_of_object) if all_filtered_points_of_object else all_filtered_points_of_object
            draw_projected_3D_points(lidar2cam, frame, point_cloud_3D, point_cloud_2D, all_filtered_points_of_object_combined)

            # Create a bev representation
            bev = create_BEV(all_filtered_points_of_object, all_pred_corners_3D)

            all_processed_frames.append(frame)
            all_bev.append(bev)

        # Create and save all the processed frames and bev as a .mp4 video
        create_combined_video(all_processed_frames, all_bev, os.path.join(output_path_video, f"processed_video_{last_video_index+1}.mp4"))

        # Optionally only save the processed frames without bev (uncomment next line)
        #create_video(all_processed_frames, os.path.join(output_path_video, f"processed_video_{last_video_index+1}.mp4"))


    elif args.image_index == "random":
        # Get the specified number of random image indices
        image_indices = [randint(0000, 7517) for _ in range(args.image_amount)]

        # Format the integers to always be 6 digits with leading zeros
        image_indices_formatted = [str(i).zfill(6) for i in image_indices]

        if len(args.output_path) == 0:
            output_path_random = "../Model_Output/Results_Random_Images/"
        else:
            output_path_random = os.path.join(args.output_path, "Results_Random_Images/")
        
        print(f"Results will be stored in {output_path_random}")
        os.makedirs(output_path_random, exist_ok=True)

        print("\n---- Processing Random Images ----")
        for index in image_indices_formatted:
            print(f"\nProcessing Image Number {index} ...")
            # Read the files specified by the argument
            image_path = os.path.join(KITTI_dataset_path, f"data_object_image_2/training/image_2/{index}.png")
            velodyne_path = os.path.join(KITTI_dataset_path, f"data_object_velodyne/training/velodyne/{index}.bin")
            calib_path = os.path.join(KITTI_dataset_path, f"data_object_calib/training/calib/{index}.txt")

            # Open the image with cv2
            frame = cv2.imread(image_path)

            # Define Lidar to camera object
            lidar2cam = LiDAR2Camera(calib_path)

            # Process the frame
            _, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, _ = detector.process_frame(frame, velodyne_path, lidar2cam, args.erosion, args.depth)

            # Draw the predicted bounding boxes onto the frame
            for pred_corner_3D in all_corners_3D:
                plot_projected_pred_bounding_boxes(lidar2cam, frame, pred_corner_3D, (0, 0, 255))

            # Draw the projected LiDAR points of the detected objects onto the frame
            draw_projected_3D_points(lidar2cam, frame, pts_3D, pts_2D, np.vstack(all_filtered_points_of_object))
                    
            # Create a bev representation
            bev = create_BEV(all_filtered_points_of_object, all_corners_3D)

            # Save the combined image for visualization
            create_combined_image(frame, bev, output_path=output_path_random + f"processed_image_{index}.png")

    else:
        print("Arguments not valid. Refer to the guidelines below for correct usage of the model.")
        parser.print_help()
