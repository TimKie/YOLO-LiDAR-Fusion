# YOLO-LiDAR Fusion
## Overview
This repository contains the code produced during my Master's Thesis in collaboration with the UBIX research group of the University of Luxembourg’s Interdisciplinary Centre for Security, Reliability, and Trust (SnT).
This thesis aimed to develop a resource-efficient model for 3D object detection utilizing LiDAR and camera sensors, tailored for autonomous vehicles with limited computational resources. An overview of the model is shown in the figure below.

![Model_Overview](assets/model_overview.svg)

## 1. Prerequisites
### Hardware
- Ideally: NVIDIA GPU such that the YOLOv8 model can be run with CUDA

### Software
- Python version between 3.8 and 3.11 (due to open3d requirements)

_Note:_ The code was developed and tested on Python 3.10. 

## 2. Setup
Follow the steps below to set up the environment:

1. Go to the folder of your choice and clone the repository:

    ```shell
    git clone https://github.com/Flavio8699/see-devops-project.git
    ```

2. Get into the working directory (root directory of the repository):

    ```shell
    cd YOLO-LiDAR-Fusion
    ```

3. Optionally start a virtual environment:

    ```python
    python -m venv venv
    ```

4. Install required libraries:

    ```python
    pip install -r requirements.txt
    ```

