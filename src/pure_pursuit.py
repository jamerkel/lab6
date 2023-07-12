#!/usr/bin/env python

import rospy
import numpy as np
import time
import utils

from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        self.lookahead = rospy.get_param("~lookahead", 1.0)
        self.speed = rospy.get_param("~speed", 0.5)
        self.wheelbase_length = rospy.get_param("~wheelbase_length", 1.0)
        self.trajectory = utils.LineTrajectory("/followed_trajectory") # provided path
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/vesc/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        self.localize_sub = rospy.Subscriber("/inferred_pose", PoseStamped, self.localize_cb, queue_size=10)
        self.current_pose = None

    def localize_cb(self, pose):

        self.current_pose = pose.pose


    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print ("Receiving new trajectory:", len(msg.poses), "points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

    def odom_callback(self, msg):
        ''' Pure pursuit control loop
        '''
        while self.current_pose == None:
            rospy.sleep()

        # Compute the closest point on the trajectory
        closest_point, closest_segment = self.get_closest_point(self.current_pose.position)

        # Compute the lookahead point
        lookahead_point = self.get_lookahead_point(closest_segment, closest_point, self.lookahead)

        # Compute the steering angle
        steering_angle = self.compute_steering_angle(lookahead_point)

        # Create and publish the AckermannDriveStamped message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.speed
        self.drive_pub.publish(drive_msg)

    def get_closest_point(self):
        ''' Compute the closest point on the trajectory to the current position
        '''
        closest_dist = float('inf')
        closest_point = None
        closest_segment = None

        for i in range(len(self.trajectory.points) - 1):
            p0 = self.trajectory.points[i]
            p1 = self.trajectory.points[i + 1]
            dist = self.distance_to_line(current_position, p0, p1)
            if dist < closest_dist:
                closest_dist = dist
                closest_point = self.get_closest_point_on_segment(self.current_pose.position, p0, p1)
                closest_segment = i

        return closest_point, closest_segment

    def distance_to_line(self, point, line_start, line_end):
        ''' Compute the distance between a point and a line defined by two points
        '''
        p0 = np.array([point.x, point.y])
        p1 = np.array([line_start[0], line_start[1]])
        p2 = np.array([line_end[0], line_end[1]])
        return np.linalg.norm(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)

    def get_closest_point_on_segment(self, point, segment_start, segment_end):
        ''' Compute the closest point on a line segment to a given point
        '''
        p = np.array([point.x, point.y])
        p1 = np.array([segment_start[0], segment_start[1]])
        p2 = np.array([segment_end[0], segment_end[1]])

        line_direction = p2 - p1
        line_length = np.linalg.norm(line_direction)
        line_direction /= line_length

        v = p - p1
        t = np.dot(v, line_direction)
        t = np.clip(t, 0, line_length)

        closest_point = p1 + line_direction * t
        return closest_point

    def get_lookahead_point(self, closest_segment, closest_point, lookahead_distance):
        ''' Compute the lookahead point on the trajectory given the closest segment and point
        '''
        lookahead_point = None

        # Iterate through the trajectory segments starting from the closest segment
        for i in range(closest_segment, len(self.points) - 1):
            segment_start = self.points[i]
            segment_end = self.points[i + 1]

            # Check if the lookahead distance exceeds the current segment length
            segment_length = np.linalg.norm(np.array(segment_end) - np.array(segment_start))
            if lookahead_distance > segment_length:
                # Reduce the lookahead distance and move to the next segment
                lookahead_distance -= segment_length
                continue

            # Compute the direction vector of the segment
            segment_direction = np.array(segment_end) - np.array(segment_start)
            segment_direction /= np.linalg.norm(segment_direction)

            # Compute the lookahead point by extending from the closest point on the segment
            lookahead_point = np.array(closest_point) + lookahead_distance * segment_direction
            break

        return lookahead_point


    def compute_steering_angle(self, lookahead_point):
        ''' Compute the steering angle based on the current position, orientation, and lookahead point
        '''
        # Convert the current orientation to yaw angle
        _, _, current_yaw = utils.tf.transformations.euler_from_quaternion([self.current_pose.orientation.x, self.current_pose.orientation.y, self.current_pose.orientation.z, self.current_pose.orientation.w])

        # Compute the angle between the current position and the lookahead point
        dx = lookahead_point[0] - self.current_pose.position.x
        dy = lookahead_point[1] - current_pose.position.y
        target_yaw = np.arctan2(dy, dx)

        # Compute the steering angle
        steering_angle = np.arctan2(2 * self.wheelbase_length * np.sin(target_yaw - current_yaw), self.lookahead)

        return steering_angle

if __name__ == "__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
