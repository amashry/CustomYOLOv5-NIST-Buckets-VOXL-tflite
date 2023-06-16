# CustomYOLOv5-NIST-Buckets-VOXL-tflite 

This repository provides a detailed documentaion for training a custom YOLOv5 object detection model for real-time identification of NIST buckets using ModalAi VOXL m500 drone. The steps include data collection, labeling, model training, testing, and deployment.

## Table of Contents
- [Getting Started](#getting-started)
- [Data Collection](#data-collection)
- [Data Labeling](#data-labeling)
- [Model Training](#model-training)
- [Model Testing Offboard](#model-testing-offboard)
- [Model Deployment](#model-deployment-onboard)
- [Ongoing Work](#ongoing-work)

## Getting Started

These instructions will guide you through the process of setting up your environment, collecting and preparing the dataset, and then using that data to train and test the YOLOv5 model. Lastly, we will discuss how to deploy the model on a VOXL1 drone.

## Data Collection

For details about data collection, including drone set-up, recording of tracking messages, and a Python script for converting ROS bag messages into image frames for the dataset, please refer to [DataCollection.md](DataCollection.md).

## Data Labeling

For information about the labeling process, including steps to manually label images using Roboflow and exporting the labeled dataset, please refer to [DataLabeling.md](DataLabeling.md).

## Model Training

For a guide on the model training process, including uploading images to Google Colab and training the model using YOLOv5, please refer to [ModelTraining.md](ModelTraining.md).

## Model Testing Offboard

For a walkthrough of the model testing process, which involves recording the output of rqt_image_view on a local machine and running the YOLOv5 ROS node, please refer to [ModelTesting.md](ModelTesting.md).

## Model Deployment Onboard

For a tutorial on the process of converting the model to a .tflite format that is compatible with voxl-tflite-server, including quantization and deployment, please refer to [ModelDeployment.md](ModelDeployment.md). Note: this process is still ongoing.

## Ongoing Work

For updates on our current work on converting the .pt model to a .tflite model that works with the voxl-tflite-server, check out [OngoingWork.md](OngoingWork.md).

## Building and Deploying VOXL Packages

For detailed instructions for editing, compiling, building, and deploying VOXL packages, please refer to [VOXLPackages.md](VOXLPackages.md).
