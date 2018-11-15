#!/usr/bin/env python
"""
ROS node for detecting people, using the Movidius Neural Compute Stick SDK
Interoperates with "ros_people_object_detection_tensorflow" by Cagatay Odabasi, 
(using sample code from that library) and Movidius NCAppZoo sample code.

This node uses a pre-trained "ssd_mobilenet_graph" with a limited number of objects.
For Sheldon, we use just use it for person detection.

For object detection, we'll run "ros_people_object_detection_tensorflow", which detects
more objects. 
"""

# ROS
import rospy
import logging
import sys
import cv2
from cob_perception_msgs.msg import Detection, DetectionArray, Rect
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
#from cob_people_object_detection_tensorflow.detector import Detector
#from cob_people_object_detection_tensorflow import utils
#from body_tracker_msgs.msg import BodyTracker

# for NCS SDK (shim from v1 to v2)
sys.path.insert(0, "/home/system/ncappzoo/ncapi2_shim") 
import mvnc_simple_api as mvnc
import numpy
import time
import csv
import os

from body_tracker_msgs.msg import BodyTracker
from geometry_msgs.msg import PointStamped, Point, PoseStamped, Pose, Pose2D

# labels AKA classes.  The class IDs returned are the indices into this list
labels = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')


# only accept classifications with 1 in the class id index.
# default is to accept all object clasifications.
# for example if object_classifications_mask[1] == 0 then
#    will ignore aeroplanes
object_classifications_mask = [1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1]

# the ssd mobilenet image width and height
NETWORK_IMAGE_WIDTH = 300
NETWORK_IMAGE_HEIGHT = 300

# the minimal score for a box to be shown
DEFAULT_INIT_MIN_SCORE = 60
min_score_percent = DEFAULT_INIT_MIN_SCORE

input_video_path = '.'
cam = None 
LIVE_CAMERA = True
FRAMES_TO_SKIP = 0  # use to experiment with CPU load, etc.

ASTRA_MINI_FOV_X =  1.047200   # (60 degrees horizontal)
ASTRA_MINI_FOV_Y = -0.863938   # (49.5 degrees vertical)

class PeopleDetectionNode(object):
    def __init__(self):
        super(PeopleDetectionNode, self).__init__()

        # init the node
        rospy.init_node('people_object_detection', anonymous=False)
        rospy.loginfo("Starting NCS People detecter...")


        # Get parameters from launch file
        network_graph_path = rospy.get_param('~network_graph_path', "")
        #confirm path to ssd_mobilenet_graph file was supplied          
        if network_graph_path:
            rospy.loginfo('Found network_graph_path: ' +  network_graph_path)
        else:
            rospy.logerr("network_graph_path param is required!")
            quit()
        # NOTE: We use the 'ncappzoo/apps/video_objects/graph', local copy stored at:
        # '.../ncs_people_detection/network_graphs/ssd_mobilenet_graph'


        # Get other parameters from YAML file
        camera_rgb_topic  = rospy.get_param("~camera_rgb_topic", "/cv_camera/image_raw")
        camera_depth_topic  = rospy.get_param("~camera_depth_topic", "")
        video_file_path = rospy.get_param("~video_file_path", "")
        self.show_cv_debug_window = False
        self.show_cv_debug_window = rospy.get_param("~show_cv_debug_window", False)


        self.incoming_image_msg = None
        self.incoming_depth_msg = None
        self.cv_image = None
        self._bridge = CvBridge()
        self.skip_frame_count = 0

        # Advertise the result of Object Detector (COB format)
        self.pub_detections = rospy.Publisher('/object_detection/detections', \
            DetectionArray, queue_size=1)
        # and the marked up image
        self.pub_detections_image = rospy.Publisher(\
            '/object_detection/detections_image', Image, queue_size=1)

        # Advertise the BodyTracker message (same as Nuitrack node)
        self.pub_body_tracker = rospy.Publisher('/body_tracker/position', \
            BodyTracker, queue_size=1)

        # configure the NCS
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

        # Get a list of ALL the sticks that are plugged in
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            rospy.logerr('*** No Movidius NCS devices found!  Exiting! ***')
            quit()

        # Pick the first stick to run the network
        device = mvnc.Device(devices[0])

        # Open the NCS
        device.OpenDevice()

        # Load graph file to memory buffer
        with open(network_graph_path, mode='rb') as f:
            graph_data = f.read()

        # allocate the Graph instance from NCAPI by passing the memory buffer
        self.ssd_mobilenet_graph = device.AllocateGraph(graph_data)

        # open the camera or video file       
        if not video_file_path or video_file_path == "no" or video_file_path == "cam":

            # Subscribe to the live video messages

            if camera_depth_topic:
                # When depth is specified, synchronize RGB and Depth frames
                # warning!  if used, but no depth camera, RGB will never show up!

                # Subscribe to approx synchronized rgb and depth frames
                self.sub_rgb = message_filters.Subscriber(camera_rgb_topic, Image)
                self.sub_depth = message_filters.Subscriber(camera_depth_topic, Image)

                # Create the message filter
                ts = message_filters.ApproximateTimeSynchronizer(\
                    [sub_rgb, sub_depth], 2, 0.9)
                ts.registerCallback(self.rgb_and_depth_callback)
                rospy.loginfo('Subscribing to SYNCHRONIZED RGB: ' + \
                camera_rgb_topic + " and Depth: " + camera_depth_topic)
                
            else:
                # no depth topic, RGB only

                self.sub_rgb = rospy.Subscriber(camera_rgb_topic,\
                    Image, self.rgb_callback, queue_size=1, buff_size=2**24)
                rospy.loginfo('Subscribing to camera_rgb_topic: ' + camera_rgb_topic)

        else:
            rospy.logwarn("READING FROM VIDEO FILE INSTEAD OF ROS MESSAGES")
            self.read_from_video(video_file_path)

        # spin
        rospy.spin()


    def preprocess_image(self, source_image):
        # create a preprocessed image from the source image that complies to the
        # network expectations and return it

        resized_image = cv2.resize(source_image, (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT))
        
        # transform values from range 0-255 to range -1.0 - 1.0
        resized_image = resized_image - 127.5
        resized_image = resized_image * 0.007843
        return resized_image



    def run_inference(self, ssd_mobilenet_graph):
        # Run an inference on the passed image
        # self.cv_image is the image on which an inference will be performed
        #    upon successful return this image will be overlayed with boxes
        #    and labels identifying the found objects within the image.
        # ssd_mobilenet_graph is the Graph object from the NCAPI which will
        #    be used to peform the inference.
        # Returns:
        #    msg (cob_perception_msgs/DetectionArray) The ROS message to be sent

        #rospy.loginfo("DBG run_inference ")
        # preprocess the image to meet nework expectations
        resized_image = self.preprocess_image(self.cv_image)

        #rospy.loginfo("DBG load tensor ")
        # Send the image to the NCS as 16 bit floats
        ssd_mobilenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

        # Get the result from the NCS
        output, userobj = ssd_mobilenet_graph.GetResult()

        #   a.	First fp16 value holds the number of valid detections = num_valid.
        #   b.	The next 6 values are unused.
        #   c.	The next (7 * num_valid) values contain the valid detections data
        #       Each group of 7 values will describe an object/box These 7 values in order.
        #       The values are:
        #         0: image_id (always 0)
        #         1: class_id (this is an index into labels)
        #         2: score (this is the probability for the class)
        #         3: box left location within image as number between 0.0 and 1.0
        #         4: box top location within image as number between 0.0 and 1.0
        #         5: box right location within image as number between 0.0 and 1.0
        #         6: box bottom location within image as number between 0.0 and 1.0


        # Process results, drawing bounding boxes on image and creating ROS message parameters

        # Initialize the COB Message
        msg = DetectionArray()
        msg.header = self.incoming_image_msg.header

        # get depth message (if any)
        cv_depth_image = None
        if self.incoming_depth_msg:
            # Convert image to numpy array
            self.cv_depth_image = self._bridge.imgmsg_to_cv2(self.incoming_depth_msg, "passthrough")


        # number of boxes returned
        num_valid_boxes = int(output[0])

        #rospy.loginfo("DBG draw boxes ")

        for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                not numpy.isfinite(output[base_index + 1]) or
                not numpy.isfinite(output[base_index + 2]) or
                not numpy.isfinite(output[base_index + 3]) or
                not numpy.isfinite(output[base_index + 4]) or
                not numpy.isfinite(output[base_index + 5]) or
                not numpy.isfinite(output[base_index + 6])):
                # boxes with non finite (inf, nan, etc) numbers must be ignored
                continue

            class_id = int(output[base_index + 1])
            if (class_id < 0):
                continue

            score = output[base_index + 2]  # Percent confidence, 0.0 --> 1.0
            percentage = int(score * 100)
            if (percentage <= min_score_percent):
                continue

            # Calculate Bounding Box position from 0,0 (top left of image)
            source_image_width = self.cv_image.shape[1]
            source_image_height = self.cv_image.shape[0]

            bb_left = max(int(output[base_index + 3] * source_image_width), 0)
            bb_top = max(int(output[base_index + 4] * source_image_height), 0)
            bb_right = min(int(output[base_index + 5] * source_image_width), source_image_width-1)
            bb_bottom = min(int(output[base_index + 6] * source_image_height), source_image_height-1)
            bb_width = bb_right - bb_left
            bb_height = bb_bottom - bb_top

            # Create the COB detection message
            detection = Detection()
            detection.header = self.incoming_image_msg.header
            detection.label = labels[class_id]
            detection.id = class_id
            detection.score = score
            detection.detector = 'Tensorflow object detector'
            detection.mask.roi.x = bb_left
            detection.mask.roi.y = bb_top
            detection.mask.roi.width = bb_width
            detection.mask.roi.height = bb_height

            # determine person top center
            body_center_x = bb_left + (bb_width / 2)
            body_center_y = bb_top + (bb_height / 2)
            body_top_y = bb_top # for tracking head
            

            # Create the Body Tracker message (same as the Nuitrack node uses)
            body_tracker_msg = BodyTracker()
            body_tracker_msg.body_id = -1
            body_tracker_msg.tracking_status = 0
            body_tracker_msg.gesture = -1 # no gesture

            # ==============================================================
            # convert to radians from center of camera
            body_position_radians_x = ((body_center_x / float(source_image_width)) - 0.5) * ASTRA_MINI_FOV_X
            body_position_radians_y = ((body_top_y / float(source_image_height)) - 0.5) * ASTRA_MINI_FOV_Y
            body_tracker_msg.position2d.x = body_position_radians_x 
            body_tracker_msg.position2d.y = body_position_radians_y 

            # 3d position relative to camera (need TF with servo position to get actual)
            body_tracker_msg.position3d.x = 0.0
            body_tracker_msg.position3d.y = 0.0
            body_tracker_msg.position3d.z = 0.0

            if cv_depth_image:
                # find average depth of person

                cv_depth_bounding_box = cv_depth[bb_top:bb_top+bb_height, \
                    bb_left:bb_left+bb_width]
                try:
                    depth_mean = numpy.nanmedian(\
                       cv_depth_bounding_box[numpy.nonzero(cv_depth_bounding_box)])

                    body_tracker_msg.position3d.x = body_position_radians_x
                    body_tracker_msg.position3d.y = body_position_radians_y
                    body_tracker_msg.position3d.z = depth_mean*0.001

                except Exception as e:
                    print e


            # ==============================================================


            # publish Body Tracker message for each person separately        
            self.pub_body_tracker.publish(body_tracker_msg)

            # Log object info
            rospy.loginfo("Found Object:  ID: " + str(class_id) + " Label: " \
            + labels[class_id] + " Confidence: " + str(percentage) + "%")

            rospy.loginfo("ROI:  x: " + str(bb_left) + " y: " + str(bb_top) + \
                " w: " + str(bb_width) + " h: " + str(bb_height))

            # overlay boxes and labels on to the image
            self.overlay_on_image(self.cv_image, output[base_index:base_index + 7])

            msg.detections.append(detection)

        return (msg)


    def overlay_on_image(self, display_image, object_info):
        # overlays the boxes and labels onto the display image.
        # display_image is the image on which to overlay the boxes/labels
        # object_info is a list of 7 values as returned from the network
        #     These 7 values describe the object found and they are:
        #         0: image_id (always 0 for myriad)
        #         1: class_id (this is an index into labels)
        #         2: score (this is the probability for the class)
        #         3: box left location within image as number between 0.0 and 1.0
        #         4: box top location within image as number between 0.0 and 1.0
        #         5: box right location within image as number between 0.0 and 1.0
        #         6: box bottom location within image as number between 0.0 and 1.0
        # returns None

        source_image_width = display_image.shape[1]
        source_image_height = display_image.shape[0]

        #rospy.loginfo("DBG overlay_on_image ")

        base_index = 0
        class_id = int(object_info[base_index + 1])
        if (class_id < 0):
            return

        #if (class_id != 15):
        #    rospy.loginfo("DBG NOT A PERSON, skipping ")
        #    return  

        if (object_classifications_mask[class_id] == 0):
            return

        percentage = int(object_info[base_index + 2] * 100)
        if (percentage <= min_score_percent):
            return

        #rospy.loginfo("DBG got object: ")
        label_text = str(class_id) + ": " + labels[class_id] + " (" + str(percentage) + "%)"
        box_left = int(object_info[base_index + 3] * source_image_width)
        box_top = int(object_info[base_index + 4] * source_image_height)
        box_right = int(object_info[base_index + 5] * source_image_width)
        box_bottom = int(object_info[base_index + 6] * source_image_height)
        #rospy.loginfo(label_text)

        box_color = (255, 128, 0)  # box color
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

        # determine color
        scale_max = (100.0 - min_score_percent)
        scaled_prob = (percentage - min_score_percent)
        scale = scaled_prob / scale_max

        # draw the classification label string just above and to the left of the rectangle
        #label_background_color = (70, 120, 70)  # greyish green background for text
        label_background_color = (0, int(scale * 175), 75)
        label_text_color = (255, 255, 255)  # white text

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        if (label_top < 1):
            label_top = 1
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                      label_background_color, -1)

        # label text above the box
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


    def read_from_video(self, video_name):
        # For testing only
        # Do object detection on a video file instead of ROS image messages

        cap = cv2.VideoCapture(video_name)

        while(cap.isOpened()):
            ret, frame = cap.read()

            if frame is not None:
                image_message = \
                    self._bridge.cv2_to_imgmsg(frame, "bgr8")

                self.rgb_callback(image_message)

            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        print "Video has been processed!"

        self.shutdown()

    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("See ya!")


    def rgb_and_depth_callback(self, rgb_msg, depth_msg):
        """
        Callback for synchronized RGB and Depth frames
        Allows distance to people to be determined 
        """

        # save the depth image
        self.incoming_depth_msg = depth_msg

        # call the rgb frame handler as usual
        rgb_callback(rgb_msg)
        


    def rgb_callback(self, data):
        """
        Callback for RGB images
        """
        if self.skip_frame_count < FRAMES_TO_SKIP:
            self.skip_frame_count += 1
            rospy.loginfo("DBG skipping frame ")
            return

        self.skip_frame_count = 0 
        #rospy.loginfo("========= DBG processing new frame ==========================")

        try:

            self.incoming_image_msg = data
            # Convert image to numpy array
            self.cv_image = self._bridge.imgmsg_to_cv2(data, "bgr8")

            # Detect and draw bounding boxes
            #rospy.loginfo("DBG run_inference ")
            msg = self.run_inference(self.ssd_mobilenet_graph)

            # local OpenCV display for testing
            if self.show_cv_debug_window:
                #rospy.loginfo("DBG cv2.imshow ")
                cv2.imshow("NCS People Detecton Node", self.cv_image)
                raw_key = cv2.waitKey(1)

            # Convert numpy image into sensor img
            msg_im = \
                self._bridge.cv2_to_imgmsg(\
                self.cv_image, encoding="passthrough")

            # Publish the COB messages
            self.pub_detections.publish(msg)
            self.pub_detections_image.publish(msg_im)



        except CvBridgeError as e:
            print(e)



def main():
    """ main function
    """
    node = PeopleDetectionNode()

if __name__ == '__main__':
    main()
