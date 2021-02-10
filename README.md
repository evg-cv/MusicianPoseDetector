# MusicianPoseDetector

## Overview

This project is to estimate the pose of musicians in the video and analyze their poses using OpenPose. PyTorch, Tensorflow, 
OpenCV libraries are used in this project.

## Structure

- src

    The main source code for pose detection and estimation.
    
- utils

    * The model for person detection
    * The source code of utilities for pose calculation and folder & file management
    
- app

    The main execution file
    
- requirements

    All the dependencies with their version
    
- settings

    Several settings including the video file path

## Installation

- Environment

    Ubuntu 18.04, Python 3.6

- Dependency Installation

    Please run the following commands in the terminal
    ```
        pip3 install numpy==1.19.5 
        pip3 install -r requirements.txt  
    ```

- Please create the "model" folder in "utils" folder and copy the model into "model" folder.

## Execution

- Please set VIDEO_PATH variable in settings file with the full path of video file to process

- Please run the following command in the terminal.

    ```
        python3 app.py
    ```
