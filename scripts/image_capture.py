'''
    This python script converts the rosbag into image frames to be used later in training 
'''
#!/usr/bin/env python
import roslib
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# for converting ROS messages to OpenCV
bridge = CvBridge()

counter = 0 

with rosbag.Bag('4/my_recording.bag', 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=['/tracking']):
        if counter % 1 == 0: 
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            image_filename = "4/frame-%06i.png" % msg.header.seq
            cv2.imwrite(image_filename, cv_image)
        counter +=1