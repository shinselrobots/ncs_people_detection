# ncs_people_detection

ROS node for detecting people, using the Movidius Neural Compute Stick SDK

Interoperates with the "ros_people_object_detection_tensorflow" node by Cagatay Odabasi,
replacing the person detection with a much less CPU intensive node, but sending the same
ROS messages, so the rest of the nodes work (person ID, object tracking, etc.)

Includes sample code from Movidius SDK / Movidius AppZoo and "ros_people_object_detection_tensorflow"

References:
- https://github.com/cagbal/ros_people_object_detection_tensorflow
- https://github.com/movidius/ncappzoo/tree/master/apps/video_objects

