#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from skimage.morphology import binary_erosion, disk
import tf
import tf.transformations


class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        #self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)
        self.start_sub = rospy.Subscriber("/initialpose", PoseWithCovariance, self.initialize_start, queue_size=1)
        self.obs_threshold = 50

    def map_cb(self, msg):
        self.res = msg.info.resolution #m/cell
        self.map_quat = msg.info.origin.orientation #rad quat
        map_pos = msg.info.origin.position #m,m
        self.translation = np.array([map_pos.x, map_pos.y, map_pos.z])
        self.h = msg.info.height #cell
        self.w = msg.info.width #cell
        self.map_header = msg.header

        data = msg.data #occupancydata list
        pre_erosion = np.array(data).reshape(self.h, self.w) #INDEXED AS v,u!!!!!
        erosion_data = self.data / 100

        #apply transformation and erosion to map, convert to m
        radius = 10
        footprint = disk(radius)
        self.data = (binary_erosion(erosion_data > self.obstacle_threshold, footprint=footprint) * 100).astype(int8)
        print("Map Initialized")


    def xy_to_uv(self, msg):
        #convert point to corresponding cell
        coords = np.array([msg.x, msg.y, msg.z])
        quat = np.array([-self.map_quat.x, -self.map_quat.y, -self.map_quat.z, self.map_quat.w])
        rotation = tf.transformations.quaternion_matrix(quat)
        rotated = tf.transformations.vector_multiply(rotation, coords)

        rotated -= self.translation
        
        transformed = rotated / self.res
        return transformed[:2].astype('int32')

    def uv_to_traj(self, path):
        #convert path cell to map coordinates
        quat = np.array([self.map_quat.x, self.map_quat.y, self.map_quat.z, self.map_quat.w])
        rotation = tf.transformations.quaternion_matrix(quat)

        path_xy = np.array([])

        for cell in path:
            cell.append(0)
            cell *= self.res
            transformed = tf.transformations.vector_multiply(rotation, cell)
            transformed += self.translation
            path_xy.append(cell)

        self.trajectory = PoseArray()
        self.trajectory.header = LineTrajectory.make_header(self.map_header.frame_id)
        for position in path_xy
            pose = PoseStamped()
            pose.header = self.trajectory.header
            pose.pose.position = position
            pose.pose.orientation.w = 1.

            self.trajectory.poses.append(pose.pose)

        self.traj_pub.publish(self.trajectory)
        

    def odom_cb(self, msg):
        self.odom = msg


    def goal_cb(self, msg):
        self.goal_pose_xy = msg.pose # has .position and .orientation
        self.goal_pos_uv = self.xy_to_uv(self.goal_pose_xy.position)
        self.goal_initialized = True

        self.plan_path()

    def initialize_start(self, msg):
        self.start_pose_xy = msg.pose # has .position and .orientation
        self.start_pos_uv = self.xy_to_uv(self.start_pose_xy.position)
        self.start_initialized = True

        self.plan_path()
        
    def plan_path(self):

        if not self.start_initialized and not self.goal_initialized:
            return

        start = self.start_pos_uv
        goal = self.goal_pos_uv
        

        queue = Queue()
        queue.put(start)

        visited = set()
        visited.add(start)

        parent = {}
        parent[start] = None

        while not queue.empty():
            current = queue.get()

            if current == goal:
                path = []
                while current is not None:
                    path.append[current]
                    current = parent[current]
                path.reverse()
                self.uv_to_traj(np.array(path))
                # publish trajectory
                #self.traj_pub.publish(self.trajectory.toPoseArray())

                # visualize trajectory Markers
                #self.trajectory.publish_viz()
                Line(self.trajectory)

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.put(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current

    def get_neighbors(self, cell):

        row, col = cell #v,u
        neighbors = []

        directions = [(0,1), (0,-1), (1,0), (-1,0)]

        for direction in directions:
            new_row = row + direction[0]
            new_col = col + direction[1]

            if 0 <= new_row < self.h and 0 <= new_col < self.w and self.data[new_row][new_col] < self.obs_threshold:
                neighbors.append([new_col, new_row]) #send as u,v instead of v,u

        return neighbors
            


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
