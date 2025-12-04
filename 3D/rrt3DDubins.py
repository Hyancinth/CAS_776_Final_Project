"""
Originally from https://github.com/FelicienC/RRT-Dubins/blob/master/code/dubins.py
3D RRT with Dubins Path Planning
Integrates 2D Dubins curves with 3D RRT for curvature-constrained path planning
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import vedo
from generate3D import generate3DArray, generateSTL

# Needed to increase recursion limit for large RRTs
import sys
sys.setrecursionlimit(50000)

# ============================================================================
# DUBINS PATH PLANNER (2D)
# ============================================================================

def ortho(vect2d):
    """Computes an orthogonal vector to the one given"""
    return np.array((-vect2d[1], vect2d[0]))

def dist(pt_a, pt_b):
    """Euclidian distance between two (x, y) points"""
    return ((pt_a[0]-pt_b[0])**2 + (pt_a[1]-pt_b[1])**2)**.5

class Dubins:
    """2D Dubins path planner with constant turn radius"""
    def __init__(self, radius, point_separation):
        assert radius > 0 and point_separation > 0
        self.radius = radius
        self.point_separation = point_separation

    def dubins_path(self, start, end):
        """Returns array of (x,y) points for shortest Dubins path"""
        options = self.all_options(start, end)
        dubins_path, straight = min(options, key=lambda x: x[0])[1:]
        return self.generate_points(start, end, dubins_path, straight)

    def all_options(self, start, end):
        """Compute all 6 Dubins path options"""
        center_0_left = self.find_center(start, 'L')
        center_0_right = self.find_center(start, 'R')
        center_2_left = self.find_center(end, 'L')
        center_2_right = self.find_center(end, 'R')
        
        options = [
            self.lsl(start, end, center_0_left, center_2_left),
            self.rsr(start, end, center_0_right, center_2_right),
            self.rsl(start, end, center_0_right, center_2_left),
            self.lsr(start, end, center_0_left, center_2_right),
            self.rlr(start, end, center_0_right, center_2_right),
            self.lrl(start, end, center_0_left, center_2_left)
        ]
        return options

    def find_center(self, point, side):
        """Find center of turn circle"""
        assert side in 'LR'
        angle = point[2] + (np.pi/2 if side == 'L' else -np.pi/2)
        return np.array((point[0] + np.cos(angle)*self.radius,
                         point[1] + np.sin(angle)*self.radius))

    def lsl(self, start, end, center_0, center_2):
        """Left-Straight-Left trajectory"""
        straight_dist = dist(center_0, center_2)
        alpha = np.arctan2((center_2-center_0)[1], (center_2-center_0)[0])
        beta_2 = (end[2]-alpha)%(2*np.pi)
        beta_0 = (alpha-start[2])%(2*np.pi)
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (beta_0, beta_2, straight_dist), True)

    def rsr(self, start, end, center_0, center_2):
        """Right-Straight-Right trajectory"""
        straight_dist = dist(center_0, center_2)
        alpha = np.arctan2((center_2-center_0)[1], (center_2-center_0)[0])
        beta_2 = (-end[2]+alpha)%(2*np.pi)
        beta_0 = (-alpha+start[2])%(2*np.pi)
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (-beta_0, -beta_2, straight_dist), True)

    def rsl(self, start, end, center_0, center_2):
        """Right-Straight-Left trajectory"""
        median_point = (center_2 - center_0)/2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self.radius:
            return (float('inf'), (0, 0, 0), True)
        alpha = np.arccos(self.radius/half_intercenter)
        beta_0 = -(psia+alpha-start[2]-np.pi/2)%(2*np.pi)
        beta_2 = (np.pi+end[2]-np.pi/2-alpha-psia)%(2*np.pi)
        straight_dist = 2*(half_intercenter**2-self.radius**2)**.5
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (-beta_0, beta_2, straight_dist), True)

    def lsr(self, start, end, center_0, center_2):
        """Left-Straight-Right trajectory"""
        median_point = (center_2 - center_0)/2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self.radius:
            return (float('inf'), (0, 0, 0), True)
        alpha = np.arccos(self.radius/half_intercenter)
        beta_0 = (psia-alpha-start[2]+np.pi/2)%(2*np.pi)
        beta_2 = (.5*np.pi-end[2]-alpha+psia)%(2*np.pi)
        straight_dist = 2*(half_intercenter**2-self.radius**2)**.5
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (beta_0, -beta_2, straight_dist), True)

    def lrl(self, start, end, center_0, center_2):
        """Left-Right-Left trajectory"""
        dist_intercenter = dist(center_0, center_2)
        intercenter = (center_2 - center_0)/2
        psia = np.arctan2(intercenter[1], intercenter[0])
        if dist_intercenter < 2*self.radius or dist_intercenter > 4*self.radius:
            return (float('inf'), (0, 0, 0), False)
        gamma = 2*np.arcsin(dist_intercenter/(4*self.radius))
        beta_0 = (psia-start[2]+np.pi/2+(np.pi-gamma)/2)%(2*np.pi)
        beta_1 = (-psia+np.pi/2+end[2]+(np.pi-gamma)/2)%(2*np.pi)
        total_len = (2*np.pi-gamma+abs(beta_0)+abs(beta_1))*self.radius
        return (total_len, (beta_0, beta_1, 2*np.pi-gamma), False)

    def rlr(self, start, end, center_0, center_2):
        """Right-Left-Right trajectory"""
        dist_intercenter = dist(center_0, center_2)
        intercenter = (center_2 - center_0)/2
        psia = np.arctan2(intercenter[1], intercenter[0])
        if dist_intercenter < 2*self.radius or dist_intercenter > 4*self.radius:
            return (float('inf'), (0, 0, 0), False)
        gamma = 2*np.arcsin(dist_intercenter/(4*self.radius))
        beta_0 = -((-psia+(start[2]+np.pi/2)+(np.pi-gamma)/2)%(2*np.pi))
        beta_1 = -((psia+np.pi/2-end[2]+(np.pi-gamma)/2)%(2*np.pi))
        total_len = (2*np.pi-gamma+abs(beta_0)+abs(beta_1))*self.radius
        return (total_len, (beta_0, beta_1, 2*np.pi-gamma), False)

    def generate_points(self, start, end, dubins_path, straight):
        """Generate point sequence along Dubins path"""
        if straight:
            return self.generate_points_straight(start, end, dubins_path)
        return self.generate_points_curve(start, end, dubins_path)

    def generate_points_straight(self, start, end, path):
        """Generate points for straight segment paths (LSL, RSR, LSR, RSL)"""
        total = self.radius*(abs(path[1])+abs(path[0]))+path[2]
        center_0 = self.find_center(start, 'L' if path[0] > 0 else 'R')
        center_2 = self.find_center(end, 'L' if path[1] > 0 else 'R')

        # Find straight segment endpoints
        if abs(path[0]) > 0:
            angle = start[2]+(abs(path[0])-np.pi/2)*np.sign(path[0])
            ini = center_0+self.radius*np.array([np.cos(angle), np.sin(angle)])
        else: 
            ini = np.array(start[:2])
        
        if abs(path[1]) > 0:
            angle = end[2]+(-abs(path[1])-np.pi/2)*np.sign(path[1])
            fin = center_2+self.radius*np.array([np.cos(angle), np.sin(angle)])
        else: 
            fin = np.array(end[:2])
        
        dist_straight = dist(ini, fin)

        points = []
        for x in np.arange(0, total, self.point_separation):
            if x < abs(path[0])*self.radius:
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1])*self.radius:
                points.append(self.circle_arc(end, path[1], center_2, x-total))
            else:
                coeff = (x-abs(path[0])*self.radius)/dist_straight
                points.append(coeff*fin + (1-coeff)*ini)
        points.append(end[:2])
        return np.array(points)

    def generate_points_curve(self, start, end, path):
        """Generate points for curved paths (LRL, RLR)"""
        total = self.radius*(abs(path[1])+abs(path[0])+abs(path[2]))
        center_0 = self.find_center(start, 'L' if path[0] > 0 else 'R')
        center_2 = self.find_center(end, 'L' if path[1] > 0 else 'R')
        intercenter = dist(center_0, center_2)
        center_1 = (center_0 + center_2)/2 + \
                   np.sign(path[0])*ortho((center_2-center_0)/intercenter) * \
                   (4*self.radius**2-(intercenter/2)**2)**.5
        psi_0 = np.arctan2((center_1 - center_0)[1],
                           (center_1 - center_0)[0])-np.pi

        points = []
        for x in np.arange(0, total, self.point_separation):
            if x < abs(path[0])*self.radius:
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1])*self.radius:
                points.append(self.circle_arc(end, path[1], center_2, x-total))
            else:
                angle = psi_0-np.sign(path[0])*(x/self.radius-abs(path[0]))
                vect = np.array([np.cos(angle), np.sin(angle)])
                points.append(center_1+self.radius*vect)
        points.append(end[:2])
        return np.array(points)

    def circle_arc(self, reference, beta, center, x):
        """Get point on circular arc"""
        angle = reference[2]+((x/self.radius)-np.pi/2)*np.sign(beta)
        vect = np.array([np.cos(angle), np.sin(angle)])
        return center+self.radius*vect

# ============================================================================
# 3D RRT WITH DUBINS CONSTRAINTS
# ============================================================================

class Nodes:
    """RRT node with position and heading"""
    def __init__(self, x, y, z, heading):
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading  # yaw angle in radians
        self.parent_x = []
        self.parent_y = []
        self.parent_z = []
        self.parent_heading = []
        self.curve_segments = []  # Store 3D curve points for visualization

def dist_3d(x1, y1, z1, x2, y2, z2):
    """3D Euclidean distance"""
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def check_collision_dubins(x1, y1, z1, heading1, x2, y2, z2, heading2,
                          img, stepSize, end, dubins_planner):
    """
    Steer from (x2,y2,z2,heading2) toward (x1,y1,z1,heading1) using Dubins curves
    Returns the curve points for visualization!
    """
    # Compute 2D Dubins path
    start_2d = (x2, y2, heading2)
    end_2d = (x1, y1, heading1)
    
    try:
        dubins_points = dubins_planner.dubins_path(start_2d, end_2d)
    except:
        # If Dubins fails, return invalid connection
        return (x2, y2, z2, heading2, False, False, None)
    
    # Calculate total path length
    path_length = 0
    for i in range(1, len(dubins_points)):
        path_length += np.linalg.norm(dubins_points[i] - dubins_points[i-1])
    
    # Limit to stepSize
    if path_length > stepSize:
        ratio = stepSize / path_length
        # Resample points to match stepSize
        target_len = 0
        limited_points = [dubins_points[0]]
        for i in range(1, len(dubins_points)):
            seg_len = np.linalg.norm(dubins_points[i] - dubins_points[i-1])
            if target_len + seg_len >= stepSize:
                # Interpolate final point
                remaining = stepSize - target_len
                t = remaining / seg_len
                final_pt = dubins_points[i-1] + t * (dubins_points[i] - dubins_points[i-1])
                limited_points.append(final_pt)
                break
            limited_points.append(dubins_points[i])
            target_len += seg_len
    else:
        limited_points = dubins_points
    
    # Interpolate z along the curve
    z_diff = z1 - z2
    num_points = len(limited_points)
    if path_length > 0:
        z_values = [z2 + z_diff * min(1.0, i / (num_points-1)) for i in range(num_points)]
    else:
        z_values = [z2] * num_points
    
    # Create 3D curve points for visualization
    curve_3d = [(limited_points[i][0], limited_points[i][1], z_values[i]) 
                for i in range(len(limited_points))]
    
    # New endpoint
    tx, ty = limited_points[-1][0], limited_points[-1][1]
    tz = z_values[-1]
    
    # Calculate new heading from last two points
    if len(limited_points) >= 2:
        dx = limited_points[-1][0] - limited_points[-2][0]
        dy = limited_points[-1][1] - limited_points[-2][1]
        new_heading = np.arctan2(dy, dx)
    else:
        new_heading = heading2
    
    # Boundary check
    hx, hy, hz = img.shape
    if tx < 0 or tx >= hx or ty < 0 or ty >= hy or tz < 0 or tz >= hz:
        return (tx, ty, tz, new_heading, False, False, None)
    
    # Collision check along entire curve
    obstacleThreshold = 230
    for i in range(len(limited_points)):
        xi = int(limited_points[i][0])
        yi = int(limited_points[i][1])
        zi = int(z_values[i])
        
        if xi < 0 or xi >= hx or yi < 0 or yi >= hy or zi < 0 or zi >= hz:
            return (tx, ty, tz, new_heading, False, False, None)
        
        if img[xi, yi, zi] < obstacleThreshold:
            return (tx, ty, tz, new_heading, False, False, None)
    
    # Check direct connection to goal (simplified - just check if close)
    dist_to_goal = dist_3d(tx, ty, tz, end[0], end[1], end[2])
    directCon = (dist_to_goal < stepSize * 2)
    nodeCon = True
    
    return (tx, ty, tz, new_heading, directCon, nodeCon, curve_3d)

def nearest_node(x, y, z, node_list):
    """Find nearest node in tree"""
    temp_dist = []
    for i in range(len(node_list)):
        if node_list[i] is not None:
            dist = dist_3d(x, y, z, node_list[i].x, node_list[i].y, node_list[i].z)
            temp_dist.append(dist)
        else:
            temp_dist.append(float('inf'))
    return temp_dist.index(min(temp_dist))

def rnd_point(hz, hy, hx, goal=None, bias_toward_goal=False):
    """Generate random point with heading (heading will be adjusted by Dubins)"""
    new_x = random.randint(0, hx-1)
    new_y = random.randint(0, hy-1)
    new_z = random.randint(0, hz-1)
    
    # Note: The heading here is just a placeholder
    # The actual heading will come from the Dubins curve direction
    new_heading = random.uniform(0, 2*np.pi)
    
    return (new_x, new_y, new_z, new_heading)

def RRT_Dubins(img, start, end, stepSize, dubins_radius=8.0, ax=None):
    """
    3D RRT with Dubins path constraints
    
    Parameters:
    - img: 3D obstacle grid
    - start: [x, y, z, heading]
    - end: [x, y, z, heading]
    - stepSize: extension distance
    - dubins_radius: minimum turn radius
    - ax: matplotlib 3D axis for visualization
    """
    hx, hy, hz = img.shape
    print(f"\n{'='*60}")
    print(f"3D RRT with Dubins Constraints")
    print(f"{'='*60}")
    print(f"Grid shape: {hx} x {hy} x {hz}")
    print(f"Dubins radius: {dubins_radius}")
    print(f"Step size: {stepSize}")
    print(f"{'='*60}\n")
    
    # Initialize Dubins planner
    dubins_planner = Dubins(radius=dubins_radius, point_separation=1.0)
    
    # Initialize tree
    node_list = []
    node_list.append(Nodes(start[0], start[1], start[2], start[3]))
    node_list[0].parent_x.append(start[0])
    node_list[0].parent_y.append(start[1])
    node_list[0].parent_z.append(start[2])
    node_list[0].parent_heading.append(start[3])
    
    # Visualization
    if ax is not None:
        ax.scatter([start[0]], [start[1]], [start[2]], color='green', s=100, marker='o', label='Start')
        ax.scatter([end[0]], [end[1]], [end[2]], color='red', s=100, marker='o', label='Goal')
    
    pathFound = False
    max_iterations = 20000
    i = 1
    iteration = 0
    
    while not pathFound and iteration < max_iterations:
        iteration += 1
        
        # Goal biasing (20% for Dubins since it's harder to connect)
        if random.random() < 0.2:
            nx, ny, nz, n_heading = end[0], end[1], end[2], end[3]
        else:
            nx, ny, nz, n_heading = rnd_point(hz, hy, hx, end)
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: {len(node_list)} nodes in tree")
        
        # Find nearest node
        nearestIndex = nearest_node(nx, ny, nz, node_list)
        nearest_x = node_list[nearestIndex].x
        nearest_y = node_list[nearestIndex].y
        nearest_z = node_list[nearestIndex].z
        nearest_heading = node_list[nearestIndex].heading
        
        # Steer with Dubins constraints
        tx, ty, tz, t_heading, directCon, nodeCon, curve_3d = check_collision_dubins(
            nx, ny, nz, n_heading,
            nearest_x, nearest_y, nearest_z, nearest_heading,
            img, stepSize, end, dubins_planner
        )
        
        if directCon and nodeCon:
            print(f"\n✓ Path found at iteration {iteration}!")
            
            # Add final node
            node_list.append(Nodes(tx, ty, tz, t_heading))
            node_list[i].parent_x = node_list[nearestIndex].parent_x.copy()
            node_list[i].parent_y = node_list[nearestIndex].parent_y.copy()
            node_list[i].parent_z = node_list[nearestIndex].parent_z.copy()
            node_list[i].parent_heading = node_list[nearestIndex].parent_heading.copy()
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)
            node_list[i].parent_z.append(tz)
            node_list[i].parent_heading.append(t_heading)
            if curve_3d:
                node_list[i].curve_segments = node_list[nearestIndex].curve_segments.copy()
                node_list[i].curve_segments.append(curve_3d)
            
            # Visualize
            if ax is not None:
                # Draw the curved segment (not straight line!)
                if curve_3d:
                    curve_array = np.array(curve_3d)
                    ax.plot(curve_array[:, 0], curve_array[:, 1], curve_array[:, 2],
                           color='blue', linewidth=2, alpha=0.8)
                
                # Draw connection to goal
                ax.plot([tx, end[0]], [ty, end[1]], [tz, end[2]],
                       color='blue', linewidth=2, linestyle='--')
                
                # Draw ALL curved segments in the final path
                for segment in node_list[i].curve_segments:
                    seg_array = np.array(segment)
                    ax.plot(seg_array[:, 0], seg_array[:, 1], seg_array[:, 2],
                           color='red', linewidth=3, label='Dubins Path')
                
                # Remove duplicate labels
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
                
                plt.savefig('RRT_3D_Dubins_final.png', dpi=150, bbox_inches='tight')
                plt.show()
            
            pathFound = True
            print(f"Path length: {len(node_list[i].parent_x)} nodes")
            break
        
        elif nodeCon:
            # Add node to tree
            node_list.append(Nodes(tx, ty, tz, t_heading))
            node_list[i].parent_x = node_list[nearestIndex].parent_x.copy()
            node_list[i].parent_y = node_list[nearestIndex].parent_y.copy()
            node_list[i].parent_z = node_list[nearestIndex].parent_z.copy()
            node_list[i].parent_heading = node_list[nearestIndex].parent_heading.copy()
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)
            node_list[i].parent_z.append(tz)
            node_list[i].parent_heading.append(t_heading)
            if curve_3d:
                node_list[i].curve_segments = node_list[nearestIndex].curve_segments.copy()
                node_list[i].curve_segments.append(curve_3d)
            
            if ax is not None and iteration % 50 == 0:
                # Draw curved segment (not straight line!)
                if curve_3d:
                    curve_array = np.array(curve_3d)
                    ax.plot(curve_array[:, 0], curve_array[:, 1], curve_array[:, 2],
                           color='lightblue', linewidth=0.5, alpha=0.3)
                plt.pause(0.001)
            
            i += 1
    
    if not pathFound:
        print(f"\n✗ Failed to find path after {max_iterations} iterations")
    
    return pathFound, node_list

def draw_cube(ax, corner, size, color='gray', alpha=0.2):
    """Draw a 3D cube obstacle"""
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
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, 
                                        linewidths=1, edgecolors='black', alpha=alpha))
node_list = []
node_list_2 = []

def sz_points_to_xyz_points(sz_points, path):
    """Convert a list of points in the sz grid back to 3D coordinates using the original path"""
    nodes = path[-1].curve_segments

    xyz_points = []

    # Get the nodes in the curve segments
    nodes_x = []
    nodes_y = []
    nodes_z = []

    for segment in nodes:
        for pt in segment:
            nodes_x.append(pt[0])
            nodes_y.append(pt[1])
            nodes_z.append(pt[2])
    
    for sz_pt in sz_points:
        s = sz_pt[0]
        z = sz_pt[1]

        # Find corresponding x,y in the original path based on arc length s
        accumulated_length = 0.0
        for i in range(len(nodes_x) - 1):
            x1, y1, z1 = nodes_x[i], nodes_y[i], nodes_z[i]
            x2, y2, z2 = nodes_x[i+1], nodes_y[i+1], nodes_z[i+1]
            
            dist_3d_pts = dist_3d(x1, y1, z1, x2, y2, z2)
            if accumulated_length + dist_3d_pts >= s:
                # Interpolate to find exact point
                ratio = (s - accumulated_length) / dist_3d_pts
                x_interp = x1 + ratio * (x2 - x1)
                y_interp = y1 + ratio * (y2 - y1)
                xyz_points.append((x_interp, y_interp, z))
                break
            accumulated_length += dist_3d_pts
    return xyz_points

def generate_sz_grid(grid, path):
    """Generate a 2D slice of the 3D grid for Dubins path checking. s is arc length, z is height."""
    nodes = path[-1].curve_segments
    arc_length = 0.0

    sz_grid = np.zeros((4000, grid.shape[2], 1))

    # Assume no obstacles so we aren't blocked by obstacles if the arc length between points exceeds 1
    sz_grid.fill(255)

    nodes_x = []
    nodes_y = []
    nodes_z = []

    # Get the nodes in the curve segments
    for segment in nodes:
        for pt in segment:
            nodes_x.append(pt[0])
            nodes_y.append(pt[1])
            nodes_z.append(pt[2])
    
    for i in range(len(nodes_x) - 1):
        x1, y1, z1 = nodes_x[i], nodes_y[i], nodes_z[i]

        # Check for obstacles above and below this point in the 3D grid and mark that in the sz grid
        hz = grid.shape[2]
        for z in range(hz):
            if grid[round(x1), round(y1), z] < 230: 
                sz_grid[round(arc_length), z, 0] = 0
        
        x2, y2, z2 = nodes_x[i+1], nodes_y[i+1], nodes_z[i+1]
        
        # Calculate distance between points
        dist_3d_pts = dist_3d(x1, y1, z1, x2, y2, z2)
        arc_length += dist_3d_pts
    
    return sz_grid, arc_length

# ============================================================================
# MAIN
# ============================================================================

j = 1
mode = 1
xyz_points = []

# Set initial camera position, target, and up vector
# camera_position = [[275, -100, -100], [19, 150, 270], [0, 0, 1]]

def handle_timer(event, end):
    global j
    global mode
    global node_list
    global node_list_2
    global xyz_points

    if mode == 1:

        if j <= len(node_list) - 1:
            plotter.add(vedo.Points([ (node_list[j].x, node_list[j].y, node_list[j].z) ], c='blue', r=4))
        
        if j == len(node_list) - 1:
            nodes_x = []
            nodes_y = []
            nodes_z = []
            
            nodes = node_list[len(node_list) - 1].curve_segments
            for segment in nodes:
                for pt in segment:
                    nodes_x.append(pt[0])
                    nodes_y.append(pt[1])
                    nodes_z.append(pt[2])
            
            # nodes_x.append(points[i+1][0])
            # nodes_y.append(points[i+1][1])
            # nodes_z.append(points[i+1][2])

            #plotter.add(vedo.Line(list(zip(nodes_x, nodes_y, nodes_z)), c='orange', lw=2))

            mode = 2

            # Convert the format of the 2nd set of RRT points
            nodes_x = []
            nodes_y = []
            nodes_z = []
            
            nodes = node_list_2
            for pt in nodes:
                nodes_x.append(pt.x)
                nodes_y.append(pt.y)
                nodes_z.append(pt.z)

            # Convert the formatted the 2nd set of RRT points to xyz coords for plotting
            xyz_points = sz_points_to_xyz_points(list(zip(nodes_x, nodes_y, nodes_z)), node_list)
            
            j = 0  # Reset j for next mode
    
    elif mode == 2:

        """ if j <= len(xyz_points) - 1:
            plotter.add(vedo.Points([ (xyz_points[j][0], xyz_points[j][1], xyz_points[j][2]) ], c='purple', r=4))
        
        if j == len(xyz_points) - 1: """
        nodes_x = []
        nodes_y = []
        nodes_z = []

        # Convert sz curve segments back to xyz points for plotting
        nodes = node_list_2[len(xyz_points) - 1].curve_segments

        for segment in nodes:
            for pt in segment:
                nodes_x.append(pt[0])
                nodes_y.append(pt[1])
                nodes_z.append(pt[2])
        
        xyz_points = sz_points_to_xyz_points(list(zip(nodes_x, nodes_y, nodes_z)), node_list)

        nodes_x = []
        nodes_y = []
        nodes_z = []

        for pt in xyz_points:
            nodes_x.append(pt[0])
            nodes_y.append(pt[1])
            nodes_z.append(pt[2])

        plotter.add(vedo.Line(list(zip(nodes_x, nodes_y, nodes_z)), c='lime', lw=2))
        plotter.remove_callback("timer")
        
    j += 1
    plotter.show()

def handle_mouse(event, txt):
    i = event.at
    pt2d = event.picked2d
    pt3d = plotter.at(i).compute_world_coordinate(pt2d, objs=plotter.get_meshes())
    txt.text(f'2D coords: {pt2d}\n3D coords: {pt3d}\n')
    plotter.add(txt)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("3D RRT WITH DUBINS PATH CONSTRAINTS")
    print("="*60)
    
    grid = np.array(generate3DArray())
    grid = grid[:, 170:300, 170:350]  # Focus on a slice of the grid
    generateSTL(grid)
    
    # Define start and end with headings (in radians)
    start = [35, 192 - 170, 242 - 170, 0.0]
    end = [45, 280 - 170, 200 - 170, np.pi / 4]
    stepSize = 2  # Change the step size
    dubins_radius = 2.0  # Radius of the Dubins path
    
    node_list = []
    
    plotter = vedo.Plotter(axes=1)

    plotter.camera.SetPosition(200, 200, 200)
    plotter.camera.SetFocalPoint(35, 220, 220)
    plotter.camera.SetViewUp(0, 0, 1)

    mesh = vedo.load("mesh/mesh_1.stl")
    mesh.alpha(0.3) # transparency
    plotter.add(mesh)
    plotter.add(vedo.Points([ (start[0], start[1], start[2]) ], c='green', r=8))
    plotter.add(vedo.Points([ (end[0], end[1], end[2]) ], c='red', r=8))

    # Set start position of camera
    # plotter.fly_to([0, 0, 0])
    # plotter.azimuth(15)
    # plotter.elevation(225)
    # plotter.roll(270)
    
    # Run RRT with Dubins
    pathFound, node_list = RRT_Dubins(grid, start, end, stepSize, dubins_radius, None)

    # Run Dubins on the sz grid to smooth the curvature in the z direction
    sz_grid, arc_length = generate_sz_grid(grid, node_list)
    sz_start = [0, start[2], 0, 0]
    sz_end = [arc_length, end[2], 0, 0]
    sz_dubins_planner, node_list_2 = RRT_Dubins(sz_grid, sz_start, sz_end, stepSize, dubins_radius, None)
    
    start_time = time.time()
    
    # Animate RRT growth
    plotter.add_callback("timer", lambda event: handle_timer(event, end))
    plotter.timer_callback("create", dt=20) # dt is animation speed in ms

    # Print current mouse position
    txt = vedo.Text2D("", s=1.4, pos='bottom-left', c='black', bg='lightyellow')
    plotter.add_callback('on_left_button_press', lambda event: handle_mouse(event, txt))

    plotter.interactive()

    print("\n" + "="*60)
    if pathFound:
        print("✓ RRT with Dubins constraints complete!")
    else:
        print("✗ Failed to find path")
    print("="*60 + "\n")
