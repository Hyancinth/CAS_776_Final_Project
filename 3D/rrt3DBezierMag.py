"""
Originally from https://github.com/nimRobotics/RRT/blob/master/rrt.py
modified by: Ethan and Ryan
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
from vedo import Mesh, Points, Line
from generate3D import generate3DArray, generateSTL
import json
import sys
sys.setrecursionlimit(10000)

calibration_curves = None

class Nodes:
    """Class to store the RRT graph"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent_x = []
        self.parent_y = []
        self.parent_z = []

import json

def load_calibration_curves(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    calibs = []
    for B_str, entry in data.items():
        B = float(B_str)
        if 'centerline_mm' not in entry:
            continue
        pts2d = np.array(entry['centerline_mm'], dtype=float)
        calibs.append((B, pts2d))
    return calibs

def collision(x1, y1, z1, x2, y2, z2, img):
    
    obstacleThreshold = 230
    numSamples = 100

    t = np.linspace(0, 1, numSamples) # denotes a line from point 1 to point 2 

    # points along the line 
    x = x1 + t * (x2 - x1) 
    y = y1 + t * (y2 - y1)
    z = z1 + t * (z2 - z1)
    
    hx, hy, hz = img.shape 
    print(hx, hy, hz)
    
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




def compute_tangents_from_waypoints(xs, ys, zs):
    """
    Given waypoint coordinates as separate lists, compute a tangent unit vector
    at each waypoint as the normalized average of incoming and outgoing segment
    directions (forward/backward difference at ends).
    Returns an (N, 3) numpy array of tangent vectors.
    """
    xs = list(xs)
    ys = list(ys)
    zs = list(zs)
    n = len(xs)

    pts = np.column_stack((xs, ys, zs)).astype(float)
    tangents = np.zeros_like(pts)

    if n == 1:
        return tangents

    for i in range(n):
        if i == 0:
            d = pts[1] - pts[0]
        elif i == n - 1:
            d = pts[-1] - pts[-2]
        else:
            d_prev = pts[i] - pts[i - 1]
            d_next = pts[i + 1] - pts[i]
            # Normalize segment directions before averaging
            if np.linalg.norm(d_prev) > 1e-6:
                d_prev = d_prev / np.linalg.norm(d_prev)
            if np.linalg.norm(d_next) > 1e-6:
                d_next = d_next / np.linalg.norm(d_next)
            d = d_prev + d_next

        norm = np.linalg.norm(d)
        if norm < 1e-6:
            # Fallback: use straight line between neighbours if possible
            if 0 < i < n - 1:
                d = pts[i + 1] - pts[i - 1]
                norm = np.linalg.norm(d)

        tangents[i] = d / (norm if norm > 1e-6 else 1.0)

    return tangents


def sample_cubic_bezier(p0, c1, c2, p3, numSamples=30):
    """
    Sample a cubic Bezier segment defined by 3D control points p0, c1, c2, p3.
    Returns (x, y, z) arrays of length numSamples.
    """
    p0 = np.asarray(p0, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    t = np.linspace(0.0, 1.0, numSamples)
    omt = 1.0 - t

    omt2 = omt * omt
    omt3 = omt2 * omt
    t2 = t * t
    t3 = t2 * t

    pts = (
        omt3[:, None] * p0 +
        3.0 * (omt2 * t)[:, None] * c1 +
        3.0 * (omt * t2)[:, None] * c2 +
        t3[:, None] * p3
    )

    return pts[:, 0], pts[:, 1], pts[:, 2]

def bezier_eval(p0, c1, c2, p3, t):
    omt = 1 - t
    return (omt**3)*p0 + 3*(omt**2)*t*c1 + 3*omt*(t**2)*c2 + (t**3)*p3

def curve_distance(seg_ctrl_pts, calib_ctrl_pts):
    p0, c1, c2, p3 = seg_ctrl_pts
    q0, q1, q2, q3 = calib_ctrl_pts

    ts = np.linspace(0, 1, 50)
    diff = 0.0
    for t in ts:
        P = bezier_eval(p0, c1, c2, p3, t)
        Q = bezier_eval(q0, q1, q2, q3, t)
        diff += np.linalg.norm(P - Q)
    return diff / len(ts)

def sample_bezier_segment_3d(ctrl_pts, n=100):
    """Return n points along the cubic Bezier defined by ctrl_pts (3D)."""
    ts = np.linspace(0.0, 1.0, n)
    p0, c1, c2, p3 = ctrl_pts
    pts = np.array([ bezier_eval(p0, c1, c2, p3, t) for t in ts ])
    return pts  # shape (n, 3)

def project_to_xy(pts3d):
    """Project 3D points to 2D by dropping z."""
    return pts3d[:, :2]

def curve_distance_2d(ptsA2d, ptsB2d):
    """Compute average Euclidean distance between two same-length 2D point arrays."""
    return np.mean(np.linalg.norm(ptsA2d - ptsB2d, axis=1))

def classify_segment(ctrl_pts_3d, calibration_curves):
    """Find which calibration curve is closest (in 2D) to ctrl_pts_3d segment."""
    seg_pts3d = sample_bezier_segment_3d(ctrl_pts_3d, n=100)
    seg2d = project_to_xy(seg_pts3d)
    bestB = None
    bestErr = float('inf')
    for Bval, calib_pts3d in calibration_curves:
        calib2d = project_to_xy(calib_pts3d)
        # if calib and seg have different number of points, resample or interpolate
        n = min(len(seg2d), len(calib2d))
        err = curve_distance_2d(seg2d[:n], calib2d[:n])
        if err < bestErr:
            bestErr = err
            bestB = Bval
    return bestB, bestErr

def map_B_to_color(B, Bmin=0.0, Bmax=20.0, cmap_name='coolwarm'):
    return vedo.color_map(B, name=cmap_name, vmin=Bmin, vmax=Bmax)

def generate_bezier_path_with_tangents(xs, ys, zs, samples_per_seg=30, alpha=0.25):
    """
    Returns:
      bx, by, bz : full sampled curve
      control_points : list of (p0, c1, c2, p3) per segment
    """
    xs = list(xs)
    ys = list(ys)
    zs = list(zs)

    assert len(xs) == len(ys) == len(zs)
    n = len(xs)

    if n < 2:
        return (np.array(xs), np.array(ys), np.array(zs), [])

    pts = np.column_stack((xs, ys, zs)).astype(float)
    tangents = compute_tangents_from_waypoints(xs, ys, zs)

    path_x = []
    path_y = []
    path_z = []
    control_points = []

    for i in range(n - 1):
        p0 = pts[i]
        p3 = pts[i + 1]
        chord = np.linalg.norm(p3 - p0)
        if chord < 1e-6:
            continue

        t0 = tangents[i]
        t1 = tangents[i + 1]

        c1 = p0 + t0 * (alpha * chord)
        c2 = p3 - t1 * (alpha * chord)

        # Store control points for visualisation
        control_points.append((p0, c1, c2, p3))

        # Sample curve
        num = samples_per_seg + 1
        x_seg, y_seg, z_seg = sample_cubic_bezier(p0, c1, c2, p3, numSamples=num)

        # avoid duplicates between segments
        if i < n - 2:
            x_seg = x_seg[:-1]
            y_seg = y_seg[:-1]
            z_seg = z_seg[:-1]

        path_x.extend(x_seg)
        path_y.extend(y_seg)
        path_z.extend(z_seg)

    return np.array(path_x), np.array(path_y), np.array(path_z), control_points




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
    # new_x = random.randint(0, hx)
    # new_y = random.randint(0, hy)
    # new_z = random.randint(0, hz)
    new_x = random.randint(3, 186)
    new_y = random.randint(125, 370)
    new_z = random.randint(78, 441)
    return (new_x, new_y, new_z)

def rnd_point_near(start, goal, sigma=100, p_center=0.5):
    center = start if random.random() < p_center else goal

    x = int(np.random.normal(center[0], sigma))
    y = int(np.random.normal(center[1], sigma))
    z = int(np.random.normal(center[2], sigma))

    x = np.clip(x, 3, 186)
    y = np.clip(y, 192, 370)
    z = np.clip(z, 78, 441)

    return (x, y, z)

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
            # nx, ny, nz = rnd_point(hx, hy, hz)
            nx, ny, nz = rnd_point_near(start, end, sigma=100, p_center=0.5)
        
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
                    bx, by, bz, control_points = generate_bezier_path_with_tangents(
                        path_x, path_y, path_z,
                        samples_per_seg=30,
                        alpha=0.25
                    )

                    # draw smooth path
                    ax.plot(bx, by, bz, color='red', linewidth=3, label='Path')

                    # --- draw control points and tangent arms ---
                    for (p0, c1, c2, p3) in control_points:
                        # control points themselves
                        ax.scatter([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                                color='magenta', s=40, marker='x')

                        # tangent arms (P0->C1 and C2->P3)
                        ax.plot([p0[0], c1[0]], [p0[1], c1[1]], [p0[2], c1[2]],
                                color='magenta', linestyle='--', linewidth=1)
                        ax.plot([c2[0], p3[0]], [c2[1], p3[1]], [c2[2], p3[2]],
                                color='magenta', linestyle='--', linewidth=1)


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
        xs = node_list[j].parent_x.copy()
        ys = node_list[j].parent_y.copy()
        zs = node_list[j].parent_z.copy()

        xs.append(end[0])
        ys.append(end[1])
        zs.append(end[2])

        #Build full tangent-continuous Bezier path
        bx, by, bz, control_points = generate_bezier_path_with_tangents(
            xs, ys, zs,
            samples_per_seg=30,
            alpha=0.25
        )
        i = 0
        for (p0, c1, c2, p3) in control_points:
            ctrl = (p0, c1, c2, p3)
            Bbest, err = classify_segment(ctrl, calibration_curves)
            Btest = i / len(control_points) * 20  # linearly from 0 to 20
            i+=1
            color = map_B_to_color(Btest)

            print(f"Segment: Bbest = {Bbest}, color = {color}")
        
            seg_pts = [ tuple(bezier_eval(p0, c1, c2, p3, t)) for t in np.linspace(0.0, 1.0, 40) ]
            plotter.add(vedo.Line(seg_pts, c=color, lw=4))

        #Draw control points and tangent arms 
        for (p0, c1, c2, p3) in control_points:
            plotter.add(vedo.Sphere(c1, r=0.5, c='blue'))
            plotter.add(vedo.Sphere(c2, r=0.5, c='blue'))

            arm1 = vedo.Line([tuple(p0), tuple(c1)])
            arm1.c('blue').lw(2)
            arm1.properties.SetLineStipplePattern(0xF0F0)
            arm1.properties.SetLineStippleRepeatFactor(1)
            plotter.add(arm1)

            arm2 = vedo.Line([tuple(c2), tuple(p3)])
            arm2.c('blue').lw(2)
            arm2.properties.SetLineStipplePattern(0xF0F0)
            arm2.properties.SetLineStippleRepeatFactor(1)
            plotter.add(arm2)

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
    # generateSTL(grid)

    start = [35, 170, 242]
    end = [45, 280, 200]
    stepSize = 10
    node_list = []

    # Load calibration (2D centerline data)
    calibration_curves = load_calibration_curves("output_scaled/calibration_full_mm.json")

    plotter = vedo.Plotter(axes=1)

    mesh = vedo.load("mesh/mesh_1.stl")
    mesh.alpha(0.1) # transparency
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
    # Add continuous colour bar for magnetic field
    vals = np.linspace(0.0, 20.0, 5)  # or more for finer bar
    pts = np.zeros((10, 3))

    plotter.show(interactive=True)
    
    if pathFound:
        print("RRT complete")
    else:
        print("RRT failed to find a path")