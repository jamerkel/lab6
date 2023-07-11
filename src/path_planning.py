#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from skimage.morphology import binary_erosion, disk
import tf
import tf.transformations
from Queue import Queue


class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)
        self.start_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initialize_start, queue_size=1)
        self.obs_threshold = 50

        self.start_initialized = False
        self.goal_initialized = False
        self.path_planned = False

    def map_cb(self, msg):
        self.res = msg.info.resolution #m/cell
        self.map_quat = msg.info.origin.orientation #rad quat
        map_pos = msg.info.origin.position #m,m
        self.translation = np.array([map_pos.x, map_pos.y, map_pos.z])
        self.h = msg.info.height #cell
        self.w = msg.info.width #cell
        self.map_header = msg.header

        data = list(msg.data) #occupancydata list
        
        pre_erosion = np.array(data, dtype=np.float32).reshape(self.h, self.w)
        mask = pre_erosion == -1.
        pre_erosion /= 100.

        #apply transformation and erosion to map, convert to m
        radius = 15
        footprint = disk(radius) #this isnt working rn
        self.data = binary_erosion(pre_erosion, selem=footprint).astype(np.float32)
        self.data *= 100.

        self.data[mask] = -1
        self.data = self.data.astype(np.int8)
                    
        print("Map Initialized")


    def xy_to_uv(self, msg):
        #convert point to corresponding cell
        coords = np.array([msg.x, msg.y, msg.z])
        print(coords)
        quat = np.array([self.map_quat.x, self.map_quat.y, self.map_quat.z, self.map_quat.w])
        rotation = tf.transformations.quaternion_matrix(quat)[:3, :3]
        rotated = rotation.dot(coords)

        rotated += self.translation
        
        transformed = rotated / self.res
        print((transformed[0], transformed[1]))
        print(self.data[int(transformed[1]), int(transformed[0])])
        
        return(tuple(transformed[:2].astype('int32')))

    def uv_to_traj(self, path):
        #convert path cell to map coordinates
        quat = np.array([self.map_quat.x, self.map_quat.y, self.map_quat.z, self.map_quat.w])
        rotation = tf.transformations.quaternion_matrix(quat)[:3, :3]

        for cell in path:
            transforming = np.array([float(cell[0]), float(cell[1]), 0.])
            transforming *= self.res
            transforming -= self.translation 
            transformed = rotation.dot(transforming)

            point = Point()
            point.x = transformed[0]
            point.y = transformed[1]

            self.trajectory.addPoint(point)
        

    def odom_cb(self, msg):
        self.odom = msg


    def goal_cb(self, msg):
        self.goal_pose_xy = msg.pose # has .position and .orientation
        self.goal_pos_uv = self.xy_to_uv(self.goal_pose_xy.position)
        self.goal_initialized = True

        self.plan_path()

    def initialize_start(self, msg):
        if not self.trajectory.empty:
            self.trajectory.clear
            self.path_planned = False
        self.start_pose_xy = msg.pose.pose # has .position and .orientation
        self.start_pos_uv = self.xy_to_uv(self.start_pose_xy.position)
        self.start_initialized = True


        self.plan_path()
        
    def plan_path(self):

        if not self.start_initialized or not self.goal_initialized:
            return

        start = self.start_pos_uv
        goal = self.goal_pos_uv
        

        queue = Queue()
        queue.put(start)

        visited = set()
        visited.add(start)

        parent = {}
        parent[start] = None
        print("planning path")

        while not queue.empty():
            current = queue.get()

            if current == goal:
                self.path_planned = True
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                self.uv_to_traj(np.array(path))
                print("path planned!")
                # publish trajectory
                self.traj_pub.publish(self.trajectory.toPoseArray())

                # visualize trajectory Markers
                self.trajectory.publish_viz()
            
            if self.path_planned == False:
                neighbors = self.get_neighbors(current)
                if neighbors == None:
                    print('No path found')
                else:
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            queue.put(neighbor)
                            visited.add(neighbor)
                            parent[neighbor] = current

    def get_neighbors(self, cell):

        col, row = cell #u, v
        neighbors = []

        directions = [(0,1), (0,-1), (1,0), (-1,0)]

        for direction in directions:
            new_row = row + direction[0]
            new_col = col + direction[1]

            if 0 <= new_row < self.h and 0 <= new_col < self.w and 0 <= self.data[new_row, new_col] < self.obs_threshold:
                neighbors.append((new_col, new_row)) #send as u,v instead of v,u

        return neighbors
            


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
