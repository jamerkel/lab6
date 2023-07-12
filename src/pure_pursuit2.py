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
        self.lookahead = rospy.get_param("~lookahead", 2.0)
        self.speed = rospy.get_param("~speed", 0.5)
        self.wheelbase_length = rospy.get_param("~wheelbase_length", 0.3)
        self.trajectory = utils.LineTrajectory("/followed_trajectory") # provided path
        self.trajectory_init = False
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/vesc/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
        self.localize_sub = rospy.Subscriber("/pf/viz/inferred_pose", PoseStamped, self.localize_cb, queue_size=1)
        self.current_pose = None
        self.traj_array = None



    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print ("Receiving new trajectory:", len(msg.poses), "points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.traj_array = np.array(self.trajectory.points)
        self.trajectory_init = True

    def localize_cb(self, pose):
        ''' Pure pursuit control loop
        '''
        self.current_pose = pose.pose
        self.current_pos = pose.pose.position
        self.current_orientation = pose.pose.orientation
        if self.trajectory_init == True:
            # Compute the closest point on the trajectory
            #closest_point, closest_segment = self.get_closest_point()

            # Compute the lookahead point
            lookahead_point = self.look_ahead()
            #self.get_lookahead_point(closest_segment, closest_point, self.lookahead)

            # send drive command
            self.drive_cmd(lookahead_point)


    def look_ahead(self):
        for i in range(len(self.traj_array) - 1):
            traj_vec = self.traj_array[i+1] - self.traj_array[i]
            robot_vec = self.traj_array[i] - np.array([self.current_pos.x, self.current_pos.y])
            a = np.dot(traj_vec, traj_vec)
            b = 2 * np.dot(traj_vec, robot_vec)
            c = np.dot(robot_vec, robot_vec) - self.lookahead**2
            discriminant = b**2 - 4*a*c

            if discriminant >= 0:

                p1 = (-b + np.sqrt(discriminant))/(2*a)
                p2 = (-b - np.sqrt(discriminant))/(2*a)

                if p1 >= 0 and p1 <= 1:
                    #success
                    return p1 * traj_vec + robot_vec
                if p2 >= 0 and p2 <= 1:
                    #success
                    return p2 * traj_vec + robot_vec
        return self.traj_array[0,:]
    '''
    def get_closest_point(self):
        ''' #Compute the closest point on the trajectory to the current position
    '''
        while not self.current_pose or not self.trajectory:
            rospy.sleep()
        closest_dist = float('inf')
        closest_point = None
        closest_segment = None

        for i in range(len(self.trajectory.points)-1):
            p0 = self.trajectory.points[i]
            p1 = self.trajectory.points[i+1]
            dist = self.distance_to_line(p0, p1)
            if dist < closest_dist:
                closest_dist = dist
                closest_point = self.get_closest_point_on_segment(p0, p1)
                closest_segment = i

        return closest_point, closest_segment

    def distance_to_line(self, line_start, line_end):
        ''' #Compute the distance between a point and a line defined by two points
    '''
        p0 = np.array([self.current_pos.x, self.current_pos.y])
        vector = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
        perp = np.array([-vector[1], vector[0]])
        perp /= np.linalg.norm(perp)
        disp = np.array([p0[0] - line_start[0], p0[1] - line_start[1]])

        dist = np.abs(np.dot(disp, perp))
        return dist

    def get_closest_point_on_segment(self, segment_start, segment_end):
        ''' #Compute the closest point on a line segment to a given point
    '''
        p = np.array([self.current_pos.x, self.current_pos.y])
        p1 = np.array([segment_start[0], segment_start[1]])
        p2 = np.array([segment_end[0], segment_end[1]])

        line_direction = p2 - p1
        line_length = np.linalg.norm(line_direction)
        line_direction /=  line_length

        v = p - p1
        t = np.dot(v, line_direction)
        t = np.clip(t, 0, line_length)

        closest_point = p1 + line_direction * t
        return closest_point

    def get_lookahead_point(self, closest_segment, closest_point, lookahead_distance):
        ''' #Compute the lookahead point on the trajectory given the closest segment and point
    '''
        lookahead_point = None

        # Iterate through the trajectory segments starting from the closest segment
        for i in range(closest_segment, len(self.trajectory.points) - 1):
            segment_start = self.trajectory.points[i]
            segment_end = self.trajectory.points[i + 1]

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

        print(lookahead_point)
        #print((lookahead_point[0] - self.pos.x, lookahead_point[1] - self.pos.y))

        return lookahead_point'''


    def drive_cmd(self, lookahead_point):
        ''' Compute the steering angle based on the current position, orientation, and lookahead point
    '''
        # Convert the current orientation to yaw angle
        #_, _, current_yaw = utils.tf.transformations.euler_from_quaternion([self.current_orientation.x, self.current_orientation.y, self.current_orientation.z, self.current_orientation.w])

        # Compute the angle between the current position and the lookahead point
        

            #dx = lookahead_point[0] - self.current_pos.x
            #dy = lookahead_point[1] - self.current_pos.y
            #target_yaw = np.arctan2(dy, dx)

            # Compute the steering angle
            #steering_angle = np.arctan2(2 * self.wheelbase_length * np.sin(target_yaw - current_yaw), self.lookahead)
        R = self.lookahead**2 / (2*lookahead_point[1])
        steering_angle = np.arctan(self.wheelbase_length / R)

        velocity = self.speed

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)


if __name__ == "__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
