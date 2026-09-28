#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np 
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError


class LineFollower(Node):


    def __init__(self):
        super().__init__('line_follower')
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(Image, '/camera/image_raw',
                                                  self.callback, 1)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)


    ## @brief Processes the camera images and publishes steering commands.
    #  @param data ROS2 Image message from the robot camera.
    def callback(self, data):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        frame_grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        _, black_white = cv2.threshold(
            frame_grey,
            100,
            255,
            cv2.THRESH_BINARY_INV
        )

        height, width = black_white.shape

        row = int(height * 0.9)

        chosen_row = black_white[row, :]

        line_positions = np.where(chosen_row == 255)[0]

        move = Twist()

        if len(line_positions) > 0:

            middle = int(np.mean(line_positions))

            image_center = width / 2

            error = middle - image_center
            Kp = 0.005

            move.linear.x = 0.2

            move.angular.z = -Kp * error

        else:
            move.linear.x = 0.0
            move.angular.z = 0.0

        self.cmd_vel_pub.publish(move)

        cv2.imshow("Camera", cv_image)
        cv2.imshow("Threshold", black_white)
        cv2.waitKey(1)


def main(args=None):
    print("Starting...")
    rclpy.init(args=args)
    lf = LineFollower()

    try:
        rclpy.spin(lf)
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        cv2.destroyAllWindows()
        lf.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()