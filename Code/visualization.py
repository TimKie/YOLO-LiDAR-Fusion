import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from utils import compute_box_3d
from calibration import *
from utils import *


def generate_colormap(num_colors=256, start_hue_angle=254):
    start_hue = start_hue_angle / 360.0
    cmap = plt.colormaps["hsv"]
    colors = cmap(np.linspace(start_hue, start_hue + 1, num_colors) % 1.0)[:, :3] * 255
    return colors.astype(np.uint8)


def assign_colors_by_depth(pts_3D):
    colors = generate_colormap(num_colors=256, start_hue_angle=254)

    max_depth = np.max(pts_3D[:, 0])
    min_depth = np.min(pts_3D[:, 0])
    depth_range = max_depth - min_depth

    normalized_depth = (pts_3D[:, 0] - min_depth) / depth_range
    indices = (normalized_depth * (len(colors) - 1)).astype(int)
    indices = np.clip(indices, 0, len(colors) - 1)

    return colors[indices]


def create_BEV(points, bboxes, resolution=0.08):
    # Determine the grid size based on the resolution
    x_min, x_max = 0, 50
    y_min, y_max = -20, 20
    x_bins = np.arange(x_min, x_max + resolution, resolution)
    y_bins = np.arange(y_min, y_max + resolution, resolution)

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 5))

    # Define the colormap here, so it's available for both conditions
    cmap = plt.colormaps["plasma"]
    bad_color = 'lightgrey'  # Define the color used for the background
    cmap.set_bad(color=bad_color)  # Set the background color for masked areas

    if len(points) != 0:
        # Project 3D points onto the ground plane (ignore z-coordinate)
        points = np.concatenate(points)
        projected_points = points[:, :2]

        # Create 2D histogram
        bev, x_edges, y_edges = np.histogram2d(projected_points[:, 0], projected_points[:, 1], bins=(x_bins, y_bins))

        # Mask the areas where there are no points
        masked_bev = np.ma.masked_where(bev == 0, bev)

        # Plot the BEV with swapped X and Y axes
        im = ax.imshow(masked_bev, origin='lower', extent=[y_edges[-1], y_edges[0], x_edges[0], x_edges[-1]], cmap=cmap)
    
        # Create a colorbar (uncomment when shwoing the whole figure with axes, etc.)
        #cb = fig.colorbar(im, label='Point density')
    else: 
        # If there are no points, return an image with only the background of the colormap
        fig.set_facecolor(bad_color)

    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')  
    ax.set_title('Bird\'s Eye View')

    # Plot bounding boxes
    for corners in bboxes:
        # Define edges of the bounding box
        edges = [    
            [corners[0], corners[1]],
            [corners[0], corners[2]],
            [corners[0], corners[3]],
            [corners[1], corners[6]],
            [corners[1], corners[7]],
            [corners[2], corners[5]],
            [corners[2], corners[7]],
            [corners[3], corners[5]],
            [corners[3], corners[6]],
            [corners[4], corners[5]],
            [corners[4], corners[7]],
            [corners[4], corners[6]],
        ] 

        # Define the edges with swapped x and y coordinates
        swapped_edges = [[[edge[0][1], edge[0][0]], [edge[1][1], edge[1][0]]] for edge in edges]

        # Plot edges of the bounding box
        for edge in swapped_edges:
            x_coords = [-edge[0][0], -edge[1][0]]
            y_coords = [edge[0][1], edge[1][1]]
            ax.plot(x_coords, y_coords, color='b', linewidth=1)

    # Draw a point at the location of the car
    ax.scatter(0, 0, c='tab:orange', marker='o', s=400)

    # Draw a line that simulates the vehicles going in a straight line
    # ax.plot([0, 0], [x_edges[0], x_edges[-1]], color='grey', linestyle='-')

    # Set the limits to make the bounding boxes overflow
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)

    # plt.savefig("../Test_Data4/test.png", bbox_inches='tight', pad_inches=0, dpi=300)

    # Save the figure without showing the axes and padding
    fig.gca().set_axis_off()  # Turn off axis
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # Remove padding
    ax.margins(0, 0)
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())

    # Render the plot to a numpy array
    fig.canvas.draw()

    # Convert the canvas to a numpy array
    bev_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert RGBA to RGB
    bev_array = bev_array[:, :, :3]

    # Close the figure to avoid accumulating plots
    plt.close(fig)  

    return bev_array


def create_BEV_gt_and_pred(points, gt_boxes, pred_boxes, resolution=0.08):
    # Determine the grid size based on the resolution
    x_min, x_max = 0, 50
    y_min, y_max = -20, 20
    x_bins = np.arange(x_min, x_max + resolution, resolution)
    y_bins = np.arange(y_min, y_max + resolution, resolution)

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 5))

    if len(points) != 0:
        # Project 3D points onto the ground plane (ignore z-coordinate)
        points = np.concatenate(points)
        projected_points = points[:, :2]

        # Create 2D histogram
        bev, x_edges, y_edges = np.histogram2d(
            projected_points[:, 0], projected_points[:, 1], bins=(x_bins, y_bins))

        # Mask the areas where there are no points
        masked_bev = np.ma.masked_where(bev == 0, bev)

        # Plot the BEV with swapped X and Y axes
        cmap = plt.colormaps["plasma"]
        cmap.set_bad(color='lightgrey')  # Set the background color

        im = ax.imshow(masked_bev, origin='lower', extent=[y_edges[-1], y_edges[0], x_edges[0], x_edges[-1]], cmap=cmap)
    
        # Create a colorbar (uncomment when shwoing the whole figure with axes, etc.)
        #cb = fig.colorbar(im, label='Point density')
    else: 
        # If there are no points, return an image with only the background of the colormap
        background_color = cmap(0)
        fig.set_facecolor(background_color)

    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')  
    ax.set_title('Bird\'s Eye View')

    # Plot bounding boxes
    for corners in gt_boxes:
        edges = []
    
        # Define the edges based on the common bounding box structure
        edges_indices = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Bottom face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        # Map the corner indices to actual points
        for (start_idx, end_idx) in edges_indices:
            edges.append((corners[start_idx], corners[end_idx]))

        # Define the edges with swapped x and y coordinates
        swapped_edges = [[[edge[0][1], edge[0][0]], [edge[1][1], edge[1][0]]] for edge in edges]

        # Plot edges of the bounding box
        for edge in swapped_edges:
            x_coords = [-edge[0][0], -edge[1][0]]
            y_coords = [edge[0][1], edge[1][1]]
            ax.plot(x_coords, y_coords, color='r', linewidth=1)

    # Plot bounding boxes
    for corners in pred_boxes:
        edges = [    
            [corners[0], corners[1]],
            [corners[0], corners[2]],
            [corners[0], corners[3]],
            [corners[1], corners[6]],
            [corners[1], corners[7]],
            [corners[2], corners[5]],
            [corners[2], corners[7]],
            [corners[3], corners[5]],
            [corners[3], corners[6]],
            [corners[4], corners[5]],
            [corners[4], corners[7]],
            [corners[4], corners[6]],
        ]

        # Define the edges with swapped x and y coordinates
        swapped_edges = [[[edge[0][1], edge[0][0]], [edge[1][1], edge[1][0]]] for edge in edges]

        # Plot edges of the bounding box
        for edge in swapped_edges:
            x_coords = [-edge[0][0], -edge[1][0]]
            y_coords = [edge[0][1], edge[1][1]]
            ax.plot(x_coords, y_coords, color='b', linewidth=1)


    # Draw a point at the location of the car
    ax.scatter(0, 0, c='tab:orange', marker='o', s=400)

    # Draw a line that simulates the vehicles going in a straight line
    # ax.plot([0, 0], [x_edges[0], x_edges[-1]], color='grey', linestyle='-')

    # Set the limits to make the bounding boxes overflow
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)

    # plt.savefig("../Test_Data4/test.png", bbox_inches='tight', pad_inches=0, dpi=300)

    # Save the figure without showing the axes and padding
    fig.gca().set_axis_off()  # Turn off axis
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # Remove padding
    ax.margins(0, 0)
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())

    # Render the plot to a numpy array
    fig.canvas.draw()

    # Convert the canvas to a numpy array
    bev_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert RGBA to RGB
    bev_array = bev_array[:, :, :3]

    # Close the figure to avoid accumulating plots
    plt.close(fig)  

    return bev_array


def create_combined_image(frame, bev, output_path):
    """ Create a combined image from a frame and bev representation """
    #print("\nCreating combined image ...")

    # Shapes of the images
    height_frame, width_frame, _ = frame.shape
    height_bev, width_bev, _ = bev.shape

    # Calculate the top and bottom padding for the frame image
    top_pad_frame = (height_bev - height_frame) // 2
    bottom_pad_frame = height_bev - height_frame - top_pad_frame

    # Pad the frame image with black pixels
    padded_frame = cv2.copyMakeBorder(frame, top_pad_frame, bottom_pad_frame, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Concatenate images horizontally
    combined_image = np.hstack((padded_frame, bev))

    if output_path == "show":
        cv2.imshow("Combined Image", combined_image)
        cv2.waitKey()
    else:
        # Save the combined image
        cv2.imwrite(output_path, combined_image)
    

def create_video(frames, output_path):
    """ Create a video from the processed images """

    height, width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10

    video = cv2.VideoWriter(output_path, fourcc, fps, (width,height))

    for frame in frames:
            video.write(frame)

    cv2.destroyAllWindows()
    video.release()

    print(f"\n Video saved successfully at {output_path}")


def pad_frame(frame, target_height):
    """ Add black bars to the frame to match the target height """

    height, width = frame.shape[:2]
    pad_height = target_height - height
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    return cv2.copyMakeBorder(frame, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def create_combined_video(images_left, images_right, output_path):
    """ Create a combined video from frames of two separate videos """

    # Check if both lists of images have the same length
    if len(images_left) != len(images_right):
        raise ValueError("Number of frames in the left and right videos must be the same.")

    # Get the dimensions of the frames
    height_left, width_left, _ = images_left[0].shape
    height_right, width_right, _ = images_right[0].shape

    # Choose the maximum height from both videos
    max_height = max(height_left, height_right)

    # Pad frames from both videos to have the same height
    padded_frames_left = [pad_frame(frame, max_height) for frame in images_left]
    padded_frames_right = [pad_frame(frame, max_height) for frame in images_right]

    # Define the output video parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    combined_width = width_left + width_right  # Combined width for both videos

    # Create the video writer
    video = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, max_height))

    # Iterate through each pair of frames and horizontally concatenate them
    for i, (frame_left, frame_right) in enumerate(zip(padded_frames_left, padded_frames_right)):
        combined_frame = np.concatenate((frame_left, frame_right), axis=1)

        # Add text overlay for frame number
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (20, 40)
        font_scale = 1
        font_color = (255, 255, 255)  # White color
        thickness = 2
        text = f'Frame: {i + 1}'
        cv2.putText(combined_frame, text, bottom_left_corner, font, font_scale, font_color, thickness)

        video.write(combined_frame)

    # Release the video writer and close any remaining windows
    cv2.destroyAllWindows()
    video.release()

    print(f"\n Video saved successfully at {output_path}")

    
def plot_3d_bounding_boxes(centers, sizes, rotations, labels=None, colors="blue"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    all_points = []

    for i, (center, size, rotation) in enumerate(zip(centers, sizes, rotations)):
        # Compute the 3D bounding box corners
        corners_3d = compute_box_3d(size, center, rotation)
        
        # Plot the edges of the bounding box
        edges = [
            [corners_3d[j], corners_3d[(j+1)%4]] for j in range(4)
        ] + [
            [corners_3d[j+4], corners_3d[(j+1)%4+4]] for j in range(4)
        ] + [
            [corners_3d[j], corners_3d[j+4]] for j in range(4)
        ]
        
        for edge in edges:
            ax.plot3D(*zip(*edge), color=colors[i])
        
        # Plot the center of the bounding box
        ax.scatter(*center, color="green")

        # Collect all points for setting equal scaling
        all_points.extend(corners_3d)
        
        # Add a label if provided
        if labels:
            ax.text(*center, labels[i], color="black")
    
    # Calculate bounds for equal scaling
    all_points = np.array(all_points)
    max_range = np.ptp(all_points, axis=0).max() / 2.0
    mid_x = np.mean(all_points[:, 0])
    mid_y = np.mean(all_points[:, 1])
    mid_z = np.mean(all_points[:, 2])
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()


def plot_projected_gt_bounding_boxes(lidar2cam, frame, gt_corners_3D, BGR_color):
    """ Plot projected 3D gt bbox onto the frame specified as input """

    gt_corners_2D = []
    for gt_corner in gt_corners_3D:
        gt_corner_2D = lidar2cam.project_3D_to_2D(gt_corner)
        gt_corners_2D.append(gt_corner_2D)

    gt_edges_2D = get_gt_bbox_edges(gt_corners_2D)
    
    for gt_edge in gt_edges_2D:
        pt1 = tuple(np.int32(gt_edge[0]))
        pt2 = tuple(np.int32(gt_edge[1]))
        cv2.line(frame, pt1, pt2, BGR_color, 2)


def plot_projected_pred_bounding_boxes(lidar2cam, frame, pred_corners_3D, BGR_color, object_ID=None):
    """ Plot projected 3D pred bbox onto the frame specified as input """

    pred_corners_2D = lidar2cam.convert_3D_to_2D(pred_corners_3D)
    pred_edges_2D = get_pred_bbox_edges(pred_corners_2D)

    for pred_edge in pred_edges_2D:
        pt1 = tuple(np.int32(pred_edge[0]))
        pt2 = tuple(np.int32(pred_edge[1]))
        cv2.line(frame, pt1, pt2, BGR_color, 2)

    if object_ID != None:
        # top corners: 1, 4, 6, 7 
        top_left_front_corner = pred_corners_2D[7]
        top_left_front_pt = (int(np.round(top_left_front_corner[0])), int(np.round(top_left_front_corner[1])) - 10)

        # Write the ID at the top-left corner
        cv2.putText(frame, f'ID: {object_ID}', top_left_front_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f'ID: {object_ID}', top_left_front_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)


def draw_projected_3D_points(lidar2cam, frame, FOV_pts_3D, FOV_pts_2D, pts_to_draw_3D):
    """ Draw desired 3D LiDAR points onto the frame specified as input """

    # If there are points in the array
    if len(pts_to_draw_3D) != 0:
        # Get color based on depth
        colors = assign_colors_by_depth(FOV_pts_3D)
        
        # Draw only the filtered points
        pts_to_draw_2D = lidar2cam.convert_3D_to_2D(np.array(pts_to_draw_3D), print_info=False)

        # Iterate over all 2D points that lie inside the FOV of the camera to get the color of each point correct
        for i in range(FOV_pts_2D.shape[0]):
            if FOV_pts_2D[i] in pts_to_draw_2D:
                color = colors[i]
                pt = (int(np.round(FOV_pts_2D[i, 0])), int(np.round(FOV_pts_2D[i, 1])))
                cv2.circle(frame, pt, 2, color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
