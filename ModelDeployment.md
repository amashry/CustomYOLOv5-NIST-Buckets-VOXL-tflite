# Model Deployment

The deployment of the model involves converting the `.pt` model into `.tflite` format compatible with `voxl-tflite-server`.

## Procedure

- The model was initially saved in a `.pt` format after the training. We are in the process of converting this model into a `.tflite` format for it to be compatible with the voxl-tflite-server, which is the server used to run tflite models on VOXL.

For more information on VOXL and its capabilities, you can refer to the [ModalAI Documentation](https://docs.modalai.com/).

# VOXL-TFLite-Server: Integrating Custom Models

This guide provides instructions on how to integrate custom models with the VOXL1 hardware using voxl-tflite-server, which serves as a hardware-accelerated TensorFlow Lite environment.

## Introduction

The voxl-tflite-server operates as a background systemd service leveraging the custom voxl-tflite package. At present, it supports two TensorFlow Lite build versions (delegates): `apq8096`, and `qrb5165`. This project employs voxl1 on an m500 drone and utilizes `apq8096-tflite`, a version of ModalAI TensorFlow Lite with the GPU delegate enabled.

As of now, six models are compatible with the VOXL1. The `yolov5` classifier is likely to be the most useful for many applications. 

## Integrating a Custom Model

To integrate a custom model, several steps are needed as described in the following sections.

### Conversion to TFLite

After you've obtained the trained model weights, you'll need to convert the model from a `.pt` (PyTorch) format to a `.tflite` format. This conversion is crucial for seamless integration with the voxl-tflite-server. This transformation requires model compression, quantization, and modification of the tflite-server code to accommodate the neural network's inputs and outputs.

### Post-training Quantization

Post-training quantization enables hardware acceleration on the GPU and ensures compatibility with the voxl-tflite-server. To perform this, you'll need to identify the input and output layer names and tensor sizes of the neural network. Tools like Netron can be helpful in visualizing this information. 

You will also need to modify the postprocessing function in the `InferenceHelper` script. The server provides several examples of this, each with a different use case and output tensor to handle. Determining the correct postprocessing function requires understanding the input/output layer tensors.

For this specific task, the network was fine-tuned using the yolov5s weights as a pre-trained model, retaining the same deep network architecture that yolov5 has. However, as voxl employs TensorFlow Lite and yolov5's custom dataset training uses PyTorch, you'll need to convert the PyTorch model (`.pt`) to a TensorFlow Lite model (`.tflite`).

To achieve this, the `TFLiteConverter` was used to convert the model to tflite. Notably, the server can run any TensorFlow Lite model using the v2.2.3 supported opsets (`TFLITE_BUILTINS` and custom opsets).

### Model Conversion with YOLOv5's Script

YOLOv5's repository contains a Python script named `export.py` that can convert a `.pt` model to a `.tflite` model. It is worth noting that the compatibility of this script with the TensorFlow Lite version running onboard on voxl is not verified at the time of this writing.

## Integration with VOXL-TFLite-Server

Once the model is ready, you can easily integrate it with voxl-tflite-server using the [`InferenceHelper`](https://gitlab.com/voxl-public/voxl-sdk/services/voxl-tflite-server/-/blob/master/include/inference_helper.h) class. Here are the files you need to modify:

- `main.cpp`: Add a post-processing function for the custom model.
- `inference_helper.cpp`
- `inference_helper.h`
- `scripts/apq8060/voxl-configure-tflite`

Feel free to explore the examples provided in the voxl-tflite-server to understand how these modifications can be made. Happy coding!

## TODO

- Successfully convert the `.pt` model to a `.tflite` model and deploy it on voxl-tflite-server.
- Test the deployed model on voxl-tflite-server for object detection.
- Update this document with the progress and results.

