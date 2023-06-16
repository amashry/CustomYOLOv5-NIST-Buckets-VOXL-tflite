CustomYOLOv5-NIST-Buckets
This repository provides a detailed guide to training a custom YOLOv5 object detection model for real-time identification of NIST buckets using drone imagery. The steps involved include data collection, labeling, model training, testing, and deployment.

Table of Contents
Getting Started
Data Collection
Data Labeling
Model Training
Model Testing
Model Deployment
Ongoing Work
Getting Started
These instructions will guide you through the process of setting up your environment, collecting and preparing the dataset, and then using that data to train and test the YOLOv5 model. Lastly, we will discuss how to deploy the model on a VOXL1 drone.

Data Collection
Details about data collection, including drone set-up, the recording of tracking messages, and a Python script for converting ROS bag messages into image frames for the dataset, can be found in the Data Collection section.

Data Labeling
The labeling process, including steps to manually label images using Roboflow and exporting the labeled dataset, can be found in the Data Labeling section.

Model Training
The model training process, including uploading images to Google Colab and training the model using YOLOv5, can be found in the Model Training section.

Model Testing
The model testing process, which involves recording the output of rqt_image_view on a local machine and running the YOLOv5 ROS node, is described in the Model Testing section.

Model Deployment
The process of converting the model to a .tflite format that is compatible with voxl-tflite-server, including quantization and deployment, can be found in the Model Deployment section. Note: this process is still ongoing.

Ongoing Work
We are currently working on converting the .pt model to a .tflite model that works with the voxl-tflite-server. Check out the Ongoing Work section for updates on our progress and challenges.

Building and Deploying VOXL Packages
Detailed instructions for editing, compiling, building, and deploying VOXL packages can be found in the VOXL Packages section.

