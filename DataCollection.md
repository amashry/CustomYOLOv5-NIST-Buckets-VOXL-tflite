# Data Collection 

The data for this project was collected using modalai m500 drone equipped with multiple camera sensors. The dataset collected here all have been collected using the tracking camera indoors in the Aerial Robotics Lab at UMD.  

## Procedure

- First, I launched the camera message topics using `voxl_mpa_to_ros`, which publishes different camera message topics such as tracking, hires, and stereo.
- To capture a diverse set of images, I moved around the buckets in different lighting conditions and bagged the `/tracking` message using the command: `$ rosbag record /tracking -O my_recording.bag`. 
- The message frames saved into a `my_recording.bag` file were then transformed into image frames using a Python script which is located in `/scripts/image_capture.py` , and these frames form the basis of the dataset.
- Dataset then has been uploaded to Roboflow to be labelled. More details about that in the Data Labeling section. 

More details about the tracking camera sensor used in voxl m500 can be found in the [M0014 Tracking Module Datasheet](https://docs.modalai.com/M0014/).
