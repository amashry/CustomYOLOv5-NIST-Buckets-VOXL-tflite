# Data Collection 

The data for this project was collected using a drone equipped with a tracking camera sensor (voxl m500). 

## Procedure

- First, I launched the camera message topics using `voxl_mpa_to_ros`, which publishes different camera message topics such as tracking, hires, and stereo.
- To capture a diverse set of images, I moved around the buckets and bagged the `/tracking` message using the command: `$ rosbag record /tracking -O my_recording.bag`. Also, I bagged the tracking camera images indoor in different lighting conditions. 
- The message frames saved into a `my_recording.bag` file were then transformed into image frames using a Python script that's located in `/scripts/image_capture.py` , and these frames form the basis of the dataset.
- Dataset then has been uploaded to Roboflow to be labelled. More details about that in the `DataLabeling.md` readme file. 

More details about the tracking camera sensor used in voxl m500 can be found in the [M0014 Tracking Module Datasheet](https://docs.modalai.com/M0014/).

## TODO

- Include more images in the dataset to improve the model's performance.
