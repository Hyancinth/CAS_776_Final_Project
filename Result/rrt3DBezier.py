"""
Originally from https://github.com/nimRobotics/RRT/blob/master/rrt.py
modified by: Ethan
"""

import cv2
import numpy as np
import math
import random
from pydicom import dcmread
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import vedo

class Nodes:
    """Class to store the RRT graph"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent_x = []
        self.parent_y = []
        self.parent_z = []

def collision(x1, y1, z1, x2, y2, z2, img):
    
    obstacleThreshold = 230
    numSamples = 100

    t = np.linspace(0, 1, numSamples) # denotes a line from point 1 to point 2 

    # points along the line 
    x = x1 + t * (x2 - x1) 
    y = y1 + t * (y2 - y1)
    z = z1 + t * (z2 - z1)
    
    hx, hy, hz = img.shape 
    
    for i in range(numSamples): 
        xi, yi, zi = int(x[i]), int(y[i]), int(z[i])
        
        # boundary check
        if (xi < 0 or xi >= hx or
            yi < 0 or yi >= hy or
            zi < 0 or zi >= hz):
            return True
        
        # obstacle check
        if img[xi, yi, zi] < obstacleThreshold:
            return True
        
    return False

def collision_bezier(x1, y1, z1, x2, y2, z2, img):
    obstacleThreshold = 230

    x, y, z = generate_bezier_points(x1, y1, z1, x2, y2, z2)
    numSamples = len(x)

    hx, hy, hz = img.shape
    for i in range(numSamples):
        xi, yi, zi = int(x[i]), int(y[i]), int(z[i])
        
        # boundary check
        if (xi < 0 or xi >= hx or
            yi < 0 or yi >= hy or
            zi < 0 or zi >= hz):
            return True
        
        # obstacle check
        if img[xi, yi, zi] < obstacleThreshold:
            return True
        
    return False

def generate_bezier_points(x1, y1, z1, x2, y2, z2):
    # x1, y1, z1 : start point
    # x2, y2, z2 : end point

    numSamples = 100

    midX = (x1+x2)/2
    midY = (y1+y2)/2
    midZ = (z1+z2)/2

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    norm = math.sqrt(dx*dx + dy*dy + dz*dz)

    if norm == 0:
        return [x1]*numSamples, [y1]*numSamples, [z1]*numSamples

    controlOffset = 0.5
    if dx and dy != 0:
        perp = (-dy/norm, dx/norm, 0)
    else:
        perp = (1, 0, 0)
    controlX = midX + perp[0] * controlOffset
    controlY = midY + perp[1] * controlOffset
    controlZ = midZ + perp[2] * controlOffset

    t = np.linspace(0, 1, numSamples)
    x = (1 - t)**2 * x1 + 2 * (1 - t) * t * controlX + t**2 * x2
    y = (1 - t)**2 * y1 + 2 * (1 - t) * t * controlY + t**2 * y2
    z = (1 - t)**2 * z1 + 2 * (1 - t) * t * controlZ + t**2 * z2

    return x, y, z

def check_collision_bezier(x1, y1, z1, x2, y2, z2, img, stepSize, end):
    dist = dist_3d(x1, y1, z1, x2, y2, z2) # distance between the two points
    if dist == 0:
        return x2, y2, z2, False, False
    # new point going in the direction of (x2, y2, z2) to (x1, y1, z1)
    x = x2 + (stepSize / dist) * (x1 - x2)
    y = y2 + (stepSize / dist) * (y1 - y2)
    z = z2 + (stepSize / dist) * (z1 - z2)

    print(f"From ({x2}, {y2}, {z2}) towards ({x1}, {y1}, {z1})")
    print(f"New point: ({x}, {y}, {z})")

    hx, hy, hz = img.shape
    # boundary check
    if x < 0 or x >= hx or y < 0 or y >= hy or z < 0 or z >= hz:
        print("Point out of bounds")
        directCon = False
        nodeCon = False
    else:
        # Check direct connection to goal
        if collision_bezier(x, y, z, end[0], end[1], end[2], img):
            directCon = False
        else:
            directCon = True

        # Check connection to nearest node
        if collision_bezier(x, y, z, x2, y2, z2, img):
            nodeCon = False
        else:
            nodeCon = True

    return (x, y, z, directCon, nodeCon)

def check_collision(x1, y1, z1, x2, y2, z2, img, stepSize, end):
    # x1,y1,z1 is the random point
    # x2,y2,z2 is the nearest node point
    # direction is from the nearest node to the random point

    dist = dist_3d(x1, y1, z1, x2, y2, z2) # distance between the two points

    # new point going in the direction of (x2, y2, z2) to (x1, y1, z1)
    x = x2 + (stepSize / dist) * (x1 - x2)
    y = y2 + (stepSize / dist) * (y1 - y2)
    z = z2 + (stepSize / dist) * (z1 - z2)
    
    print(f"From ({x2}, {y2}, {z2}) towards ({x1}, {y1}, {z1})")
    print(f"New point: ({x}, {y}, {z})")
    
    hx, hy, hz = img.shape

    # boundary check
    if x < 0 or x >= hx or y < 0 or y >= hy or z < 0 or z >= hz:
        print("Point out of bounds")
        directCon = False
        nodeCon = False
    else:
        # Check direct connection to goal
        if collision(x, y, z, end[0], end[1], end[2], img):
            directCon = False
        else:
            directCon = True
    
        # Check connection to nearest node
        if collision(x, y, z, x2, y2, z2, img):
            nodeCon = False
        else:
            nodeCon = True

    return (x, y, z, directCon, nodeCon)

def dist_3d(x1, y1, z1, x2, y2, z2):
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2) + ((z1 - z2) ** 2))

def nearest_node(x, y, z, node_list):
    # return the index of the nearest node in node_list to the point (x, y, z)
    temp_dist = []
    for i in range(len(node_list)):
        if node_list[i] is not None:
            dist = dist_3d(x, y, z, node_list[i].x, node_list[i].y, node_list[i].z)
            temp_dist.append(dist)
        else:
            temp_dist.append(float('inf'))
    return temp_dist.index(min(temp_dist)) 

def rnd_point(hx, hy, hz):
    # new_x = random.randint(0, hx)
    # new_y = random.randint(0, hy)
    # new_z = random.randint(0, hz)
    new_x = random.randint(3, 186)
    new_y = random.randint(192, 370)
    new_z = random.randint(78, 441)
    return (new_x, new_y, new_z)

def rnd_point_near(start, goal, sigma=100, p_center=0.5):
    # Select whether to bias near start or goal
    center = start if random.random() < p_center else goal

    x = int(np.random.normal(center[0], sigma))
    y = int(np.random.normal(center[1], sigma))
    z = int(np.random.normal(center[2], sigma))

    # clamp
    x = np.clip(x, 3, 186)
    y = np.clip(y, 192, 370)
    z = np.clip(z, 78, 441)

    return (x, y, z)

def bezier_curve_length(x1, y1, z1, x2, y2, z2):
    x, y, z = generate_bezier_points(x1, y1, z1, x2, y2, z2)
    totalLength = 0
    for i in range(1, len(x)):
        totalLength += dist_3d(x[i-1], y[i-1], z[i-1], x[i], y[i], z[i])
    return totalLength

def RRT_Bezier(img, start, end, stepSize, ax=None, bezier=False):
    node_list = []
    startTime = time.perf_counter()
    totalDistance = 0

    hx, hy, hz = img.shape
    print(f"Grid shape: {hx} x {hy} x {hz}")
    
    node_list.append(Nodes(start[0], start[1], start[2]))
    node_list[0].parent_x.append(start[0])
    node_list[0].parent_y.append(start[1])
    node_list[0].parent_z.append(start[2])
    
    if ax is not None:
        ax.scatter([start[0]], [start[1]], [start[2]], color='green', s=100, marker='o', label='Start')
        ax.scatter([end[0]], [end[1]], [end[2]], color='red', s=100, marker='o', label='Goal')

    pathFound = False
    max_iterations = 10000
    i = 1
    iteration = 0
    
    while not pathFound and iteration < max_iterations:
        iteration += 1
        
        # Goal biasing - 10% of time sample near goal
        if random.random() < 0.1:
            nx, ny, nz = end[0], end[1], end[2]
        else:
            # nx, ny, nz = rnd_point(hx, hy, hz)
            nx, ny, nz = rnd_point_near(start, end, sigma=30, p_center=0.5)
        
        print(f"\nIteration {iteration}: Random point: ({nx}, {ny}, {nz})")
        
        nearestIndex = nearest_node(nx, ny, nz, node_list)
        nearest_x = node_list[nearestIndex].x
        nearest_y = node_list[nearestIndex].y
        nearest_z = node_list[nearestIndex].z
        print(f"Nearest node: ({nearest_x}, {nearest_y}, {nearest_z})")
        
        if bezier:
            tx, ty, tz, directCon, nodeCon = check_collision_bezier(nx, ny, nz, nearest_x, nearest_y, nearest_z, img, stepSize, end)
            btx, bty, btz = generate_bezier_points(nearest_x, nearest_y, nearest_z, tx, ty, tz)
        else:
            tx, ty, tz, directCon, nodeCon = check_collision(nx, ny, nz, nearest_x, nearest_y, nearest_z, img, stepSize, end)
        
        if directCon and nodeCon:
            print("Direct connection to goal possible")
            totalDistance += bezier_curve_length(tx, ty, tz, end[0], end[1], end[2]) if bezier else dist_3d(tx, ty, tz, end[0], end[1], end[2])

            node_list.append(Nodes(tx, ty, tz))
            node_list[i].parent_x = node_list[nearestIndex].parent_x.copy()
            node_list[i].parent_y = node_list[nearestIndex].parent_y.copy()
            node_list[i].parent_z = node_list[nearestIndex].parent_z.copy()
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)
            node_list[i].parent_z.append(tz)

             # Draw complete path
            path_x = node_list[i].parent_x
            path_y = node_list[i].parent_y
            path_z = node_list[i].parent_z

            if bezier:
                for j in range(len(path_x)-1):
                    totalDistance += bezier_curve_length(path_x[j], path_y[j], path_z[j],
                                                            path_x[j+1], path_y[j+1], path_z[j+1])
                  
            
            pathFound = True
            print(f"RRT Path length: {len(node_list[i].parent_x)} nodes")
            break
        
        elif nodeCon:
            print("Node connected to tree")
            # totalDistance += bezier_curve_length(tx, ty, tz, nearest_x, nearest_y, nearest_z) if bezier else dist_3d(tx, ty, tz, nearest_x, nearest_y, nearest_z)

            node_list.append(Nodes(tx, ty, tz))
            node_list[i].parent_x = node_list[nearestIndex].parent_x.copy()
            node_list[i].parent_y = node_list[nearestIndex].parent_y.copy()
            node_list[i].parent_z = node_list[nearestIndex].parent_z.copy()
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)
            node_list[i].parent_z.append(tz)

            if ax is not None: 
                # ax.plot([nearest_x, tx], [nearest_y, ty], [nearest_z, tz], 
                #        color='lightblue', linewidth=0.5, alpha=0.5)
                ax.scatter([tx], [ty], [tz], color='blue', s=5)
                plt.pause(0.001)
            
            i += 1
        else:
            print("No connection possible, trying again")
            continue
    
    if not pathFound:
        print(f"Failed to find path after {max_iterations} iterations")
    
    endTime = time.perf_counter()
    totalTime = endTime - startTime
    return pathFound, node_list, totalDistance, totalTime, i, len(node_list)