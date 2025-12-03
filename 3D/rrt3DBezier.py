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
from generate3D import generate3DArray, generateSTL

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

def nearest_node(x, y, z):
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
    new_x = random.randint(0, hx)
    new_y = random.randint(0, hy)
    new_z = random.randint(0, hz)
    return (new_x, new_y, new_z)


def RRT(img, start, end, stepSize, node_list, ax=None, bezier=False):
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
    max_iterations = 5000
    i = 1
    iteration = 0
    
    while not pathFound and iteration < max_iterations:
        iteration += 1
        
        # Goal biasing - 10% of time sample near goal
        if random.random() < 0.1:
            nx, ny, nz = end[0], end[1], end[2]
        else:
            nx, ny, nz = rnd_point(hx, hy, hz)
        
        print(f"\nIteration {iteration}: Random point: ({nx}, {ny}, {nz})")
        
        nearestIndex = nearest_node(nx, ny, nz)
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
            
            node_list.append(Nodes(tx, ty, tz))
            node_list[i].parent_x = node_list[nearestIndex].parent_x.copy()
            node_list[i].parent_y = node_list[nearestIndex].parent_y.copy()
            node_list[i].parent_z = node_list[nearestIndex].parent_z.copy()
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)
            node_list[i].parent_z.append(tz)

            if ax is not None:
                # Draw final connection
                if bezier:
                    ax.plot(btx, bty, btz, color='blue', linewidth=1)
                    bEndX, bEndY, bEndZ = generate_bezier_points(tx, ty, tz, end[0], end[1], end[2])
                    ax.plot(bEndX, bEndY, bEndZ, 
                        color='blue', linewidth=2)
                
                else:
                    ax.plot([nearest_x, tx], [nearest_y, ty], [nearest_z, tz], 
                        color='blue', linewidth=1)
                    ax.plot([tx, end[0]], [ty, end[1]], [tz, end[2]], 
                        color='blue', linewidth=2)
                
                ax.scatter([tx], [ty], [tz], color='blue', s=5)
                # Draw complete path
                path_x = node_list[i].parent_x
                path_y = node_list[i].parent_y
                path_z = node_list[i].parent_z

                if bezier:
                    bezier_path_x = []
                    bezier_path_y = []
                    bezier_path_z = []
                    for j in range(len(path_x)-1):
                        bpx, bpy, bpz = generate_bezier_points(path_x[j], path_y[j], path_z[j],
                                                              path_x[j+1], path_y[j+1], path_z[j+1])
                        bezier_path_x.extend(bpx)
                        bezier_path_y.extend(bpy)
                        bezier_path_z.extend(bpz)

                    ax.plot(bezier_path_x, bezier_path_y, bezier_path_z, color='red', linewidth=3, label='Path')

                else:
                    ax.plot(path_x, path_y, path_z, color='red', linewidth=3, label='Path')
                
                ax.legend()
                if bezier:
                    plt.savefig('RRT_3D_Bezier_final.png', dpi=150, bbox_inches='tight')
                else:
                    plt.savefig('RRT_3D_final.png', dpi=150, bbox_inches='tight')
                plt.show()
            
            pathFound = True
            print(f"RRT Path length: {len(node_list[i].parent_x)} nodes")
            break
        
        elif nodeCon:
            print("Node connected to tree")
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
    
    return pathFound, node_list


def draw_cube(ax, corner, size, color='gray', alpha=0.2):
    x, y, z = corner
    dx, dy, dz = size
    vertices = [
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ]
    faces = [
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [7, 6, 2, 3]],
        [vertices[j] for j in [0, 3, 7, 4]],
        [vertices[j] for j in [1, 2, 6, 5]],
        [vertices[j] for j in [0, 1, 2, 3]],
        [vertices[j] for j in [4, 5, 6, 7]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='black', alpha=alpha))


node_list = []
j = 1

# Set initial camera position, target, and up vector
# camera_position = [[275, -100, -100], [19, 150, 270], [0, 0, 1]]

def handle_timer(event, end):
    global j

    if j <= len(node_list) - 1:
        plotter.add(vedo.Points([ (node_list[j].x, node_list[j].y, node_list[j].z) ], c='blue', r=4))
    
    if j == len(node_list) - 1:
        points = list(zip(node_list[j].parent_x, node_list[j].parent_y, node_list[j].parent_z))
        points.append((end[0], end[1], end[2]))

        for j in range(len(points) - 1):

            bezier_path_x = []
            bezier_path_y = []
            bezier_path_z = []
            bpx, bpy, bpz = generate_bezier_points(points[j][0], points[j][1], points[j][2],
                                                    points[j+1][0], points[j+1][1], points[j+1][2])
            bezier_path_x.extend(bpx)
            bezier_path_y.extend(bpy)
            bezier_path_z.extend(bpz)

            plotter.add(vedo.Line(list(zip(bezier_path_x, bezier_path_y, bezier_path_z)), c='orange', lw=2))
        
        plotter.remove_callback("timer")
        
    j += 1
    plotter.show()

def handle_mouse(event, txt):
    i = event.at
    pt2d = event.picked2d
    pt3d = plotter.at(i).compute_world_coordinate(pt2d, objs=plotter.get_meshes())
    txt.text(f'2D coords: {pt2d}\n3D coords: {pt3d}\n')
    print(f'2D coords: {pt2d}, 3D coords: {pt3d}')
    plotter.add(txt)

if __name__ == '__main__':
    # Everything's faster in numpy
    grid = np.array(generate3DArray())
    generateSTL(grid)

    start = [19, 150, 242]
    end = [45, 370, 242]
    stepSize = 2
    node_list = []
    
    plotter = vedo.Plotter(axes=1)

    mesh = vedo.load("mesh/mesh_1.stl")
    mesh.alpha(1) # transparency
    plotter.add(mesh)
    plotter.add(vedo.Points([ (start[0], start[1], start[2]) ], c='green', r=8))
    plotter.add(vedo.Points([ (end[0], end[1], end[2]) ], c='red', r=8))

    # Set start position of camera
    # plotter.fly_to([0, 0, 0])
    # plotter.azimuth(15)
    # plotter.elevation(225)
    # plotter.roll(270)

    pathFound, node_list = RRT(grid, start, end, stepSize, node_list)

    start_time = time.time()
    
    # Animate RRT growth
    plotter.add_callback("timer", lambda event: handle_timer(event, end))
    plotter.timer_callback("create", dt=20) # dt is animation speed in ms

    # Print current mouse position
    txt = vedo.Text2D("", s=1.4, pos='bottom-left', c='black', bg='lightyellow')
    plotter.add_callback('on_left_button_press', lambda event: handle_mouse(event, txt))

    plotter.interactive()
    
    if pathFound:
        print("RRT complete")
    else:
        print("RRT failed to find a path")