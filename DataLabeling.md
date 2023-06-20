# Data Labeling 

The dataset created was labeled using Roboflow.

## Procedure

- I uploaded the images to Roboflow and manually labeled every single image for eight different classes representing the NIST buckets: `['bucket-1', 'bucket-1a', "bucket-2", 'bucket-2a', 'bucket-3', 'bucket-3a', 'bucket-4', 'bucket-4a']`.
- The labeled dataset, which contains over 1500 images, is publicly available and can be found at this [link](https://universe.roboflow.com/nist-autonomous-inspection/nist-automation-buckets-detection/dataset/1). Dataset could be exported in different NN archeticture. 
- We used Roboflow's API key to integrate the labeled dataset into a Google Colab notebook for model training. The google colab notebook is also included in the scripts folder. 

For a more detailed guide on how to label your custom dataset using Roboflow, refer to this [tutorial](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/).
