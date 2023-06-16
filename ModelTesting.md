# Model Testing Offbard

Model testing was carried out on a local machine.

## Procedure

- We recorded the output of the `rqt_image_view` running on our local machine. This machine was running the `yolov5_ros` node, which in turn was subscribing to the tracking camera output. Yolov5_ros is a ros wrapper for yolov5, which i adapted from the following [repo](https://github.com/mats-robotics/yolov5_ros)
- We used two ROS nodes for this process: `voxl_mpa_to_ros.launch` running on voxl and `indoor.launch` running on the docker container on the drone.

## Comments

- There is a noticable latency in the output from the yolov5_ros node when viewed on local machine. 
