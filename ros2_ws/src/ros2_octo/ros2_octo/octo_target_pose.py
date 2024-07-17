import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import cv_bridge
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL 

import ros2_octo.octo_inference as octo_inference
import camera

# /home/jonathan/Thesis/ROS2_octo/ros2_ws/src/install/ros2_octo/lib/python3.10/site-packages:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/home/jonathan/miniconda3/envs/octo/lib/python3.10/site-packages


class OctoTargetPose(Node):

    def __init__(self):

        super().__init__('rt_target_pose_publisher')
        self.pose_publisher = self.create_publisher(Pose, 'target_pose', 10)
        self.grip_publisher = self.create_publisher(Float32, 'target_grip', 10)

        self.octo_inferer = octo_inference("Pick up the yellow banana.")

        self.cam = camera.Camera()

        self.cur_x = 0.0
        self.cur_y = 0.5
        self.cur_z = 0.5
        self.cur_roll = 0.0
        self.cur_pitch = 0.0
        self.cur_yaw = 90.0
        self.cur_grip = 0.02

        self.img_converter = cv_bridge.CvBridge()

        self.prev_image = None

        self.run_inference()

    def run_inference(self):
        self.init_target_pose()
        time.sleep(5)
        print('TOOK ON INIT POSE. RUNNING INFERENCE...')
        steps = 0
        while steps < 1000:

            # take picture
            image = self.cam.get_picture()

            images = np.stack(image, self.prev_image)[None]

            action = self.octo_inferer.run_inference(images)

            # self.publish_target_pose_deltas(action)

            print(hash(str(action)))
            print('Action: ', action)

            time.sleep(5)
            steps += 1
            print(f'Step {steps} done.')

        print('DONE RUNNING INFERENCE.')

    def publish_target_pose_deltas(self, action):

        gripper_closedness_action = action["gripper_closedness_action"]
        rotation_delta = action["rotation_delta"]
        terminate_episode = action["terminate_episode"]
        world_vector = action["world_vector"]

        self.cur_x += float(world_vector[0])
        self.cur_y += float(world_vector[1])
        self.cur_z += float(world_vector[2])
        self.cur_roll += float(rotation_delta[0])
        self.cur_pitch += float(rotation_delta[1])
        self.cur_yaw += float(rotation_delta[2])
        self.cur_grip = float(gripper_closedness_action[0])

        self.cur_x = min(max(self.cur_x, -0.5), 0.5)
        self.cur_y = min(max(self.cur_y, 0.2), 0.7)
        self.cur_z = min(max(self.cur_z, 0.2), 0.6)
        self.cur_roll = min(max(self.cur_roll, 0.0), 90.0)
        self.cur_pitch = min(max(self.cur_pitch, 0.0), 90.0)
        self.cur_yaw = min(max(self.cur_yaw, -10.0), 170.0)
        self.cur_grip = min(max(self.cur_grip, 0.02), 0.08)

        self.get_logger().info(f'Publishing target pose and grip...')

        pose_msg = Pose()
        pose_msg.position.x = self.cur_x
        pose_msg.position.y = self.cur_y
        pose_msg.position.z = self.cur_z
        pose_msg.orientation.x = self.cur_yaw
        pose_msg.orientation.y = self.cur_pitch
        pose_msg.orientation.z = self.cur_roll
        pose_msg.orientation.w = 1.0

        grip_msg = Float32()
        grip_msg.data = self.cur_grip
        print('Y: ', self.cur_y)

        self.pose_publisher.publish(pose_msg)
        self.grip_publisher.publish(grip_msg)

        self.pose_history.append([self.cur_x, self.cur_y, self.cur_z, self.cur_roll, self.cur_pitch, self.cur_yaw, self.cur_grip, terminate_episode])

    def init_target_pose(self):
        pose_msg = Pose()
        pose_msg.position.x = self.cur_x
        pose_msg.position.y = self.cur_y
        pose_msg.position.z = self.cur_z
        pose_msg.orientation.x = self.cur_yaw
        pose_msg.orientation.y = self.cur_pitch
        pose_msg.orientation.z = self.cur_roll
        pose_msg.orientation.w = 1.0

        grip_msg = Float32()
        grip_msg.data = self.cur_grip

        self.pose_publisher.publish(pose_msg)
        self.grip_publisher.publish(grip_msg)

def main(args=None):
    rclpy.init(args=args)

    rt_target_pose = OctoTargetPose()

    rclpy.spin(rt_target_pose)