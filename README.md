

https://github.com/amashry/CustomYOLOv5-NIST-Buckets-VOXL-tflite/assets/98168605/48543f6c-ba4f-4ea3-bda9-def619d64454

# CustomYOLOv5-NIST-Buckets-VOXL-tflite 

This repository provides a detailed documentaion for training a custom YOLOv5 object detection model for real-time identification of NIST buckets using ModalAi VOXL m500 drone. The steps include data collection, labeling, model training, testing, and deployment.

## Table of Contents
- [Getting Started](#getting-started)
- [Data Collection](#data-collection)
- [Data Labeling](#data-labeling)
- [Model Training](#model-training)
- [Model Testing Offboard](#model-testing-offboard)
- [Model Deployment](#model-deployment-onboard)

## Getting Started

These instructions will guide you through the process of setting up your environment, collecting and preparing the dataset, and then using that data to train and test the YOLOv5 model. Lastly, we will discuss how to deploy the model on a VOXL1 drone.

## Data Collection

For details about data collection, including drone set-up, recording of tracking messages, and a Python script for converting ROS bag messages into image frames for the dataset, please refer to [DataCollection.md](DataCollection.md). The following is a sample image from the collected dataset. 
![frame-001132](https://github.com/amashry/CustomYOLOv5-NIST-Buckets-VOXL-tflite/assets/98168605/d18d7986-1653-409f-af81-1abe2ec1a716)


## Data Labeling

For information about the labeling process, including steps to manually label images using Roboflow and exporting the labeled dataset, please refer to [DataLabeling.md](DataLabeling.md). 
![Screenshot 2023-06-03 12_51_36](https://github.com/amashry/CustomYOLOv5-NIST-Buckets-VOXL-tflite/assets/98168605/120f18d7-71cc-4325-a734-6df243d710ac)

## Model Training

For a guide on the model training process, including uploading images to Google Colab and training the model using YOLOv5, please refer to [ModelTraining.md](ModelTraining.md).

## Model Testing Offboard

For a walkthrough of the model testing process, which involves recording the output of rqt_image_view on a local machine and running the YOLOv5 ROS node, please refer to [ModelTesting.md](ModelTesting.md).

![ezgif com-video-to-gif](https://github.com/amashry/CustomYOLOv5-NIST-Buckets-VOXL-tflite/assets/98168605/5b930562-386d-4910-bc39-9ca6ffb906f8)


## Model Deployment Onboard

For a tutorial on the process of converting the model to a .tflite format that is compatible with voxl-tflite-server, including quantization and deployment, please refer to [ModelDeployment.md](ModelDeployment.md). Note: this process is still ongoing.


## Building and Deploying VOXL Packages

For detailed instructions for editing, compiling, building, and deploying VOXL packages, please refer to [VOXLPackages.md](VOXLPackages.md).

![trial_2_Manual_Inspection_Onboard_Detection_AdobeExpress](https://github.com/amashry/CustomYOLOv5-NIST-Buckets-VOXL-tflite/assets/98168605/6dda916d-a70a-4a22-8afd-9c39313d0b38)
