# YOLO-LiDAR Fusion
## 1. Overview
This repository contains the code produced during my Master's Thesis in collaboration with the UBIX research group of the University of Luxembourg’s Interdisciplinary Centre for Security, Reliability, and Trust (SnT).
This thesis aimed to develop a resource-efficient model for 3D object detection utilizing LiDAR and camera sensors, tailored for autonomous vehicles with limited computational resources. An overview of the model is shown in the figure below.

![Model_Overview](assets/model_overview.svg)

## 2. Prerequisites
### Hardware
- Ideally: NVIDIA GPU such that the YOLOv8 model can be run with CUDA

_Note:_ The model can also be run on the CPU (slower). 

### Software
- Python version between 3.8 and 3.11 (due to open3d requirements)

_Note:_ The code was developed and tested on Python 3.10. 

### File Structure
The file structure is important to use the model without modifying the dataset paths in the main.py file. It should be as follows:

_Notes:_ 

The **KITTI_dataset** directory contains the training dataset of KITTI that can be downloaded [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
The **KITTI_raw_data** directory contains raw data of consecutive frames (for video inference) of the KITTI dataset that can be downloaded [here](https://www.cvlibs.net/datasets/kitti/raw_data.php).
   
    .
    ├── assets
    │   └── ...
    ├── Code
    │   ├── calibration.py
    │   ├── data_processing.py
    │   ├── detector.py
    │   ├── evaluation.py
    │   ├── fusion.py
    │   ├── main.py
    │   ├── utils.py
    │   └── visualization.py
    ├── KITTI_dataset
    │   ├── data_object_calib
    │   │   └── training
    │   │       └── calib
    │   │           └── ...
    │   ├── data_object_image_2
    │   │   └── training
    │   │       └── image_2
    │   │           └── ...
    │   ├── data_object_label_2
    │   │   └── training
    │   │       └── label_2
    │   │           └── ...
    │   └── data_object_velodyne
    │       └── training
    │           └── velodyne
    │               └── ...
    ├── KITTI_raw_data
    │   ├── calib
    │   │   ├── calib_cam_to_cam.txt
    │   │   ├── calib_imu_to_velo.txt
    │   │   └── calib_velo_to_cam.txt
    │   ├── image_02
    │   │   ├── data
    │   │   │   └── ...
    │   │   └── timestamps.txt
    │   └── velodyne_points
    │       ├── data
    │       │   └── ...
    │       └── timestamps.txt
    └── requirements.py

## 3. Setup
Follow the steps below to set up the environment:

1. Go to the diretory of your choice and clone the repository:

    ```shell
    git clone https://github.com/TimKie/YOLO-LiDAR-Fusion.git
    ```

2. Get into the working directory (root directory of the repository):

    ```shell
    cd YOLO-LiDAR-Fusion
    ```

3. Optionally create and start a virtual environment:

    ```python
    python -m venv venv
    source venv/bin/activate
    ```

4. Install required libraries:

    ```python
    pip install -r requirements.txt
    ```

## 4. Usage
Follow the steps below to use the model:

_Note:_ Make sure that the file structure is as stated above in [Prerequisites](#2-prerequisites).

1. Go to the directory where the implementation is located:

   ```shell
   cd Code
   ```

2. 
