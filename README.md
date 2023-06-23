# CustomYOLOv5-NIST-Buckets-VOXL-tflite 

This repository provides detailed documentation for training a custom YOLOv5 object detection model for real-time identification of NIST buckets using the ModalAi VOXL m500 drone. The process is outlined in steps that include data collection, labeling, model training, off-board testing, and deployment and on-board testing on the drone.

## Table of Contents
- [Getting Started](#getting-started)
- [Data Collection](#data-collection)
- [Data Labeling](#data-labeling)
- [Model Training](#model-training)
- [Model Testing Offboard](#model-testing-offboard)
- [Model Deployment to VOXL-TFLITE-Server](#model-deployment-to-voxl-tflite-server)
- [References](#references)

## Getting Started

These instructions will guide you through the process of setting up your environment, collecting and preparing the dataset, and then using that data to train and test the YOLOv5 model. Lastly, I will discuss how to deploy the model on a VOXL1 drone.

## Data Collection

The data for this project was collected using a ModalAI m500 drone equipped with multiple camera sensors. All the datasets collected were done indoors in the Aerial Robotics Lab at UMD using the tracking camera.  

### Procedure

- I launched the camera message topics using `voxl_mpa_to_ros`, which publishes different camera message topics such as tracking, hires, and stereo.
- To capture a diverse set of images, I moved around the buckets under different lighting conditions and stored the `/tracking` message using the command: `$ rosbag record /tracking -O my_recording.bag`. 
- The message frames saved into a `my_recording.bag` file were then transformed into image frames using a Python script located in `/scripts/image_capture.py`, and these frames form the dataset.
- The dataset was then uploaded to Roboflow to be labeled. More details about this are provided in the Data Labeling section. 

More details about the tracking camera sensor used in the VOXL m500 can be found in the [M0014 Tracking Module Datasheet](https://docs.modalai.com/M0014/).

<p align="center">
  <i> A sample image from the collected dataset</i><br>
  <img src="https://github.com/amashry/CustomYOLOv5-NIST-Buckets-VOXL-tflite/assets/98168605/251f85a8-1448-44c1-abdd-300508255391"> 
</p>


## Data Labeling
The dataset was labeled using Roboflow.

### Procedure

- I uploaded the images to Roboflow and manually labeled every single image for eight different classes, representing the following NIST buckets: `['bucket-1', 'bucket-1a', "bucket-2", 'bucket-2a', 'bucket-3', 'bucket-3a', 'bucket-4', 'bucket-4a']`.
- The labeled dataset, which contains over 1500 images, is publicly available and can be found at this [link](https://universe.roboflow.com/nist-autonomous-inspection/nist-automation-buckets-detection/dataset/1). The dataset can be downloaded and exported in multiple popular formats, such as Yolov8, Yolov5, Yolov7, TFRecord, and more.
- The only preprocessing performed on the images before training was resizing them to 416x416. No augmentations were applied.
- I used Roboflow's API key to integrate the labeled dataset into a Google Colab notebook for model training. The google colab notebook is also included in the scripts folder. 

For a more detailed guide on how to label and train your custom dataset using Roboflow, refer to this [tutorial](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/).

<p align="center">
  <img src="https://github.com/amashry/CustomYOLOv5-NIST-Buckets-VOXL-tflite/assets/98168605/56ae8c06-70de-4742-9635-730ca9247497">
  <i> Labels and Train/Test split from Roboflow </i><br>
</p>


## Model Training

This project utilized Google Colab's computational resources, leveraging its free GPU for training a custom YOLOv5s model on a unique dataset.

### Procedure

1. The YOLOv5 repository was cloned onto a Google Colab notebook, which made the required files and architecture readily accessible.
2. YOLOv5s, a smaller variant of the YOLOv5 model optimized for speed and size, was selected as the pre-trained model. It was trained on batches of 16 images, each of size 416x416, for a total of 100 epochs.
3. Throughout the training process, the model's performance metrics were monitored and displayed directly in the Colab notebook.
4. Upon completion of training, the model that demonstrated the best performance (saved as `best.pt`) was selected. This model was subsequently used for detecting objects in test images, with results demonstrated within the notebook.
5. Leveraging YOLOv5's `export.py` feature, the trained model was exported in three formats: PyTorch model (.pt), TensorFlow's Frozen Graph format (.pb), and TensorFlow Lite (.tflite). These files are located in the `model` directory of this repository.

For a detailed view of the training process, refer to this [Google Colaboratory notebook](https://colab.research.google.com/drive/1heIi2KDMOVZ5hTXWi4UCEMjEXJJT38gl?usp=sharing). The notebook is also included in the `scripts` folder within this repository.

## Model Testing Offboard

To ascertain the model's applicability in the autonomy loop, we executed a real-time inference testing, starting offboard on a local machine. The model was challenged to perform object detection on a camera stream sourced directly from the drone.

### Procedure

1. The `yolov5_ros` node, a ROS wrapper for YOLOv5 adapted from this [repository](https://github.com/mats-robotics/yolov5_ros), was launched on a local machine. This node subscribed to the tracking camera output.
2. Customizations were made to the launch file to ensure the use of the correct model weights (`weights`) and data file (`data.yaml`). Additionally, the `input_image_topic` was adjusted to correspond to the intended camera topic.
3. The testing process involved two ROS nodes running on the drone: `voxl_mpa_to_ros` running on VOXL and `inspection_indoor` running on the docker container onboard the drone.
4. The output of the `rqt_image_view`, running locally, was recorded to visually document the result of the YOLOv5 ROS node's detection.

The following video presents the YOLOv5 ROS node's detection output while the drone autonomously inspected multiple buckets:

[YOLOv5 ROS node detection output video](https://github.com/amashry/CustomYOLOv5-NIST-Buckets-VOXL-tflite/assets/98168605/95a95214-47b8-478d-9901-5057a63cf8e2)


## Model Deployment to VOXL-TFLITE-Server

This section describes how to convert the `.pt` model into the `.tflite` format, suitable for the `voxl-tflite-server`. It provides guidelines for integrating custom models with VOXL1 hardware, leveraging voxl-tflite-server as a hardware-accelerated TensorFlow Lite environment. More information on VOXL and its capabilities can be found in the [ModalAI Documentation](https://docs.modalai.com/).

### Procedure

The `voxl-tflite-server` operates as a systemd background service using the custom `voxl-tflite` package. Currently, it supports two TensorFlow Lite build versions (delegates): `apq8096`, and `qrb5165`. This project uses the `apq8096-tflite` version with the GPU delegate enabled on the m500 drone employing VOXL1. 

As of the time of writing, six models are compatible with VOXL1, with the `yolov5` classifier being of most utility for many applications. 

### Conversion to TFLite

The trained model weights must be converted from `.pt` (PyTorch) format to `.tflite` format for smooth integration with the voxl-tflite-server. This conversion necessitates model compression, quantization, and modification of the tflite-server code to accommodate the neural network's input and output.

### Post-training Quantization

Post-training quantization enables hardware acceleration on the GPU and ensures compatibility with the voxl-tflite-server. You will need to identify the input and output layer names and tensor sizes of the neural network for this process. Tools like Netron can help visualize this information. 

The `InferenceHelper` script postprocessing function must also be modified. The source code contains several examples for various use cases and output tensors. Correct postprocessing function identification requires understanding the input/output layer tensors.

In this specific project, the yolov5s weights were used as a pre-trained model for network fine-tuning, maintaining the same deep network architecture. However, the PyTorch model (`.pt`) must be converted to a TensorFlow Lite model (`.tflite`) due to voxl's reliance on TensorFlow Lite and yolov5's custom dataset training using PyTorch. The `TFLiteConverter` was employed for this purpose.

The server can run any TensorFlow Lite model using the v2.2.3 supported opsets (`TFLITE_BUILTINS` and custom opsets). A Python script to perform this conversion is available in the ModalAI documentation and included in this repository under the `scripts` directory. 

### Model Conversion with YOLOv5's Python Script

YOLOv5's repository provides a Python script, `export.py`, to export the `.pt` model to a `.tflite` model. Compatibility with the TensorFlow Lite version running on voxl is not always guaranteed. If the exported tflite model does not work with the tflite server running on voxl, consider retraining the model using the TensorFlow version matching your voxl version.

### Integration with VOXL-TFLite-Server

After preparing the model, it can be easily integrated with voxl-tflite-server using the [`InferenceHelper`](https://gitlab.com/voxl-public/voxl-sdk/services/voxl-tflite-server/-/blob/master/include/inference_helper.h) class. The following files require modification:

- `main.cpp`
- `inference_helper.cpp`
- `inference_helper.h`
- `scripts/apq8060/voxl-configure-tflite`

Refer to the `voxl-tflite-server` directory to review the changes made in the above scripts for the custom model's integration. Simply search for `nist` within these scripts to find the adjustments.

After modifying the source code, follow the steps in the readme file within the `voxl-tflite-server` directory as instructed by ModalAI. I've also added the compiled `.ipk` package after modifying, building, and packaging the source code. If you wish to clone the same functionality onto your voxl drone, download the compiled `.ipk` package, push it to your drone, and install it using `opkg install voxl-tflite-server_0.3.1.ipk --force-reinstall`. 

https://github.com/amashry/CustomYOLOv5-NIST-Buckets-VOXL-tflite/assets/98168605/2d0f1383-8129-486f-8e4c-2c7c05f8915a


_The above video demonstrates the system in action. The drone autonomously inspects the buckets, runs onboard detection, and streams the output on the VOXL portal (right). It also shows offboard detection running on my local machine (middle), highlighting the time difference between onboard and offboard detection._

# References 

1. [ModalAI](https://docs.modalai.com/)
2. [yolov5](https://github.com/ultralytics/yolov5)
3. [yolov5_ros](https://github.com/mats-robotics/yolov5_ros)
4. [Roboflow](https://roboflow.com/) 
