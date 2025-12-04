"""
3D RRT with Dubins Path Planning
Reference code: https://github.com/FelicienC/RRT-Dubins
Edited by: Colton
"""

import numpy as np
import math
import random
import vedo
from generate3D import generate3DArray, generateSTL


# -----------------------------------------
# PROPER DUBINS PATH PLANNER (2D)
# -----------------------------------------

def ortho(vect2d):
    """Computes an orthogonal vector to the one given"""
    return np.array((-vect2d[1], vect2d[0]))


def dist(pt_a, pt_b):
    """Euclidian distance between two (x, y) points"""
    return ((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2) ** .5


def mod2pi(theta):
    """Normalize angle to [-pi, pi]"""
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta < -np.pi:
        theta += 2 * np.pi
    return theta


class Dubins:
    """Proper Dubins path planner with constant turn radius"""

    def __init__(self, radius, point_separation):
        assert radius > 0 and point_separation > 0
        self.radius = radius
        self.point_separation = point_separation

    def find_center(self, point, side):
        """Find center of turn circle"""
        assert side in 'LR'
        angle = point[2] + (np.pi / 2 if side == 'L' else -np.pi / 2)
        return np.array((point[0] + np.cos(angle) * self.radius,
                         point[1] + np.sin(angle) * self.radius))

    def lsl(self, start, end, center_0, center_2):
        """Left-Straight-Left"""
        straight_dist = dist(center_0, center_2)
        alpha = np.arctan2((center_2 - center_0)[1], (center_2 - center_0)[0])
        beta_2 = (end[2] - alpha) % (2 * np.pi)
        beta_0 = (alpha - start[2]) % (2 * np.pi)
        total_len = self.radius * (beta_2 + beta_0) + straight_dist
        return (total_len, (beta_0, beta_2, straight_dist), True)

    def rsr(self, start, end, center_0, center_2):
        """Right-Straight-Right"""
        straight_dist = dist(center_0, center_2)
        alpha = np.arctan2((center_2 - center_0)[1], (center_2 - center_0)[0])
        beta_2 = (-end[2] + alpha) % (2 * np.pi)
        beta_0 = (-alpha + start[2]) % (2 * np.pi)
        total_len = self.radius * (beta_2 + beta_0) + straight_dist
        return (total_len, (-beta_0, -beta_2, straight_dist), True)

    def rsl(self, start, end, center_0, center_2):
        """Right-Straight-Left"""
        median_point = (center_2 - center_0) / 2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self.radius:
            return (float('inf'), (0, 0, 0), True)
        alpha = np.arccos(self.radius / half_intercenter)
        beta_0 = -(psia + alpha - start[2] - np.pi / 2) % (2 * np.pi)
        beta_2 = (np.pi + end[2] - np.pi / 2 - alpha - psia) % (2 * np.pi)
        straight_dist = 2 * (half_intercenter ** 2 - self.radius ** 2) ** .5
        total_len = self.radius * (beta_2 + beta_0) + straight_dist
        return (total_len, (-beta_0, beta_2, straight_dist), True)

    def lsr(self, start, end, center_0, center_2):
        """Left-Straight-Right"""
        median_point = (center_2 - center_0) / 2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self.radius:
            return (float('inf'), (0, 0, 0), True)
        alpha = np.arccos(self.radius / half_intercenter)
        beta_0 = (psia - alpha - start[2] + np.pi / 2) % (2 * np.pi)
        beta_2 = (.5 * np.pi - end[2] - alpha + psia) % (2 * np.pi)
        straight_dist = 2 * (half_intercenter ** 2 - self.radius ** 2) ** .5
        total_len = self.radius * (beta_2 + beta_0) + straight_dist
        return (total_len, (beta_0, -beta_2, straight_dist), True)

    def lrl(self, start, end, center_0, center_2):
        """Left-Right-Left"""
        dist_intercenter = dist(center_0, center_2)
        intercenter = (center_2 - center_0) / 2
        psia = np.arctan2(intercenter[1], intercenter[0])
        if dist_intercenter < 2 * self.radius or dist_intercenter > 4 * self.radius:
            return (float('inf'), (0, 0, 0), False)
        gamma = 2 * np.arcsin(dist_intercenter / (4 * self.radius))
        beta_0 = (psia - start[2] + np.pi / 2 + (np.pi - gamma) / 2) % (2 * np.pi)
        beta_1 = (-psia + np.pi / 2 + end[2] + (np.pi - gamma) / 2) % (2 * np.pi)
        total_len = (2 * np.pi - gamma + abs(beta_0) + abs(beta_1)) * self.radius
        return (total_len, (beta_0, beta_1, 2 * np.pi - gamma), False)

    def rlr(self, start, end, center_0, center_2):
        """Right-Left-Right"""
        dist_intercenter = dist(center_0, center_2)
        intercenter = (center_2 - center_0) / 2
        psia = np.arctan2(intercenter[1], intercenter[0])
        if dist_intercenter < 2 * self.radius or dist_intercenter > 4 * self.radius:
            return (float('inf'), (0, 0, 0), False)
        gamma = 2 * np.arcsin(dist_intercenter / (4 * self.radius))
        beta_0 = -((-psia + (start[2] + np.pi / 2) + (np.pi - gamma) / 2) % (2 * np.pi))
        beta_1 = -((psia + np.pi / 2 - end[2] + (np.pi - gamma) / 2) % (2 * np.pi))
        total_len = (2 * np.pi - gamma + abs(beta_0) + abs(beta_1)) * self.radius
        return (total_len, (beta_0, beta_1, 2 * np.pi - gamma), False)

    def circle_arc(self, reference, beta, center, x):
        """Point on circle arc"""
        angle = reference[2] + ((x / self.radius) - np.pi / 2) * np.sign(beta)
        vect = np.array([np.cos(angle), np.sin(angle)])
        return center + self.radius * vect

    def generate_points_straight(self, start, end, path):
        """Generate points for LSL, RSR, LSR, RSL paths"""
        total = self.radius * (abs(path[1]) + abs(path[0])) + path[2]
        center_0 = self.find_center(start, 'L' if path[0] > 0 else 'R')
        center_2 = self.find_center(end, 'L' if path[1] > 0 else 'R')

        # Find straight segment endpoints
        if abs(path[0]) > 0:
            angle = start[2] + (abs(path[0]) - np.pi / 2) * np.sign(path[0])
            ini = center_0 + self.radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            ini = np.array(start[:2])

        if abs(path[1]) > 0:
            angle = end[2] + (-abs(path[1]) - np.pi / 2) * np.sign(path[1])
            fin = center_2 + self.radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            fin = np.array(end[:2])

        dist_straight = dist(ini, fin)

        # Generate points
        points = []
        for x in np.arange(0, total, self.point_separation):
            if x < abs(path[0]) * self.radius:  # First turn
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1]) * self.radius:  # Last turn
                points.append(self.circle_arc(end, path[1], center_2, x - total))
            else:  # Straight segment
                coeff = (x - abs(path[0]) * self.radius) / dist_straight
                points.append(coeff * fin + (1 - coeff) * ini)
        points.append(end[:2])
        return np.array(points)

    def generate_points_curve(self, start, end, path):
        total = self.radius * (abs(path[1]) + abs(path[0]) + abs(path[2]))
        center_0 = self.find_center(start, 'L' if path[0] > 0 else 'R')
        center_2 = self.find_center(end, 'L' if path[1] > 0 else 'R')
        intercenter = dist(center_0, center_2)
        center_1 = (center_0 + center_2) / 2 + \
                   np.sign(path[0]) * ortho((center_2 - center_0) / intercenter) \
                   * (4 * self.radius ** 2 - (intercenter / 2) ** 2) ** .5
        psi_0 = np.arctan2((center_1 - center_0)[1],
                           (center_1 - center_0)[0]) - np.pi

        points = []
        for x in np.arange(0, total, self.point_separation):
            if x < abs(path[0]) * self.radius:  # First turn
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1]) * self.radius:  # Last turn
                points.append(self.circle_arc(end, path[1], center_2, x - total))
            else:  # Middle turn
                angle = psi_0 - np.sign(path[0]) * (x / self.radius - abs(path[0]))
                vect = np.array([np.cos(angle), np.sin(angle)])
                points.append(center_1 + self.radius * vect)
        points.append(end[:2])
        return np.array(points)

    def dubins_path(self, start, end):
        """Compute shortest Dubins path"""
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

        dubins_path, straight = min(options, key=lambda x: x[0])[1:]

        if straight:
            return self.generate_points_straight(start, end, dubins_path)
        else:
            return self.generate_points_curve(start, end, dubins_path)


# ------------------------------
# 3D RRT WITH DUBINS CONSTRAINTS
# ------------------------------

class Nodes:
    def __init__(self, x, y, z, heading=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading
        self.parent_x = []
        self.parent_y = []
        self.parent_z = []
        self.parent_index = None
        self.curve_to_parent = None  # Store the smooth Dubins curve to parent


def dist_3d(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def check_collision_dubins(x1, y1, z1, heading1,
                           x2, y2, z2, heading2,
                           img, stepSize, end, dubins_planner):
    """
    Check collision along Dubins path from parent (x2,y2,z2) to target (x1,y1,z1)
    """
    # Calculate direction from parent to target
    dx = x1 - x2
    dy = y1 - y2
    distance_2d = math.sqrt(dx * dx + dy * dy)

    if distance_2d < 0.1:
        return x2, y2, z2, heading2, False, False, None

    target_heading = math.atan2(dy, dx)

    # Use parent's heading as start, target direction as end
    start_2d = (x2, y2, heading2)
    end_2d = (x1, y1, target_heading)

    # Calculate proper Dubins path
    try:
        dubins_points = dubins_planner.dubins_path(start_2d, end_2d)
    except:
        return x2, y2, z2, heading2, False, False, None

    if len(dubins_points) < 2:
        return x2, y2, z2, heading2, False, False, None

    # Calculate path length
    segment_lengths = np.linalg.norm(np.diff(dubins_points, axis=0), axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    # Limit to stepSize
    if total_length > stepSize:
        idx = np.searchsorted(cumulative_length, stepSize)
        if idx >= len(dubins_points):
            idx = len(dubins_points) - 1

        if idx > 0 and idx < len(dubins_points):
            remaining = stepSize - cumulative_length[idx - 1]
            segment_len = segment_lengths[idx - 1] if idx - 1 < len(segment_lengths) else 0
            if segment_len > 0:
                t = remaining / segment_len
                final_pt = dubins_points[idx - 1] + t * (dubins_points[idx] - dubins_points[idx - 1])
            else:
                final_pt = dubins_points[idx - 1]
            dubins_points = np.vstack([dubins_points[:idx], final_pt])

    # Z interpolation
    z_diff = z1 - z2
    num_points = len(dubins_points)
    z_values = np.linspace(z2, z2 + z_diff * min(1.0, total_length / stepSize if total_length > 0 else 1.0), num_points)

    # Create 3D curve - ensure it starts at parent position
    curve_3d = [(float(dubins_points[i][0]), float(dubins_points[i][1]), float(z_values[i]))
                for i in range(num_points)]

    # Force first point to be exactly at parent
    if len(curve_3d) > 0:
        curve_3d[0] = (float(x2), float(y2), float(z2))

    tx, ty, tz = curve_3d[-1]

    # Calculate heading from the last two points of the actual curve
    if len(curve_3d) >= 2:
        dx_curve = curve_3d[-1][0] - curve_3d[-2][0]
        dy_curve = curve_3d[-1][1] - curve_3d[-2][1]
        new_heading = math.atan2(dy_curve, dx_curve)
    elif len(dubins_points) >= 2:
        dx_curve = dubins_points[-1][0] - dubins_points[-2][0]
        dy_curve = dubins_points[-1][1] - dubins_points[-2][1]
        new_heading = math.atan2(dy_curve, dx_curve)
    else:
        new_heading = target_heading

    hx, hy, hz = img.shape

    if tx < 0 or tx >= hx or ty < 0 or ty >= hy or tz < 0 or tz >= hz:
        return tx, ty, tz, new_heading, False, False, None

    obstacleThreshold = 230

    # Collision checking
    for i in range(len(dubins_points)):
        xi, yi, zi = int(dubins_points[i][0]), int(dubins_points[i][1]), int(z_values[i])

        if xi < 0 or xi >= hx or yi < 0 or yi >= hy or zi < 0 or zi >= hz:
            return tx, ty, tz, new_heading, False, False, None

        if img[xi, yi, zi] < obstacleThreshold:
            return tx, ty, tz, new_heading, False, False, None

    # Check between points
    for i in range(len(dubins_points) - 1):
        x_start, y_start = dubins_points[i]
        x_end, y_end = dubins_points[i + 1]
        z_start, z_end = z_values[i], z_values[i + 1]

        segment_dist = np.linalg.norm([x_end - x_start, y_end - y_start, z_end - z_start])
        num_samples = max(3, int(segment_dist / 0.5))

        for j in range(num_samples):
            t = j / num_samples
            xi = int(x_start + t * (x_end - x_start))
            yi = int(y_start + t * (y_end - y_start))
            zi = int(z_start + t * (z_end - z_start))

            if xi < 0 or xi >= hx or yi < 0 or yi >= hy or zi < 0 or zi >= hz:
                return tx, ty, tz, new_heading, False, False, None

            if img[xi, yi, zi] < obstacleThreshold:
                return tx, ty, tz, new_heading, False, False, None

    dist_to_goal = dist_3d(tx, ty, tz, end[0], end[1], end[2])
    directCon = (dist_to_goal < stepSize * 2)
    nodeCon = True

    return tx, ty, tz, new_heading, directCon, nodeCon, curve_3d


def nearest_node(x, y, z, node_list):
    dists = [dist_3d(x, y, z, n.x, n.y, n.z) for n in node_list]
    return dists.index(min(dists))


def rnd_point(hx, hy, hz, start=None, end=None, bias_region=True):
    """Generate random point with optional biasing"""
    if bias_region and start is not None and end is not None:
        min_x = max(0, min(start[0], end[0]) - 20)
        max_x = min(hx - 1, max(start[0], end[0]) + 20)
        min_y = max(0, min(start[1], end[1]) - 20)
        max_y = min(hy - 1, max(start[1], end[1]) + 20)
        min_z = max(0, min(start[2], end[2]) - 20)
        max_z = min(hz - 1, max(start[2], end[2]) + 20)

        return (random.randint(int(min_x), int(max_x)),
                random.randint(int(min_y), int(max_y)),
                random.randint(int(min_z), int(max_z)),
                random.uniform(0, 2 * np.pi))
    else:
        return (random.randint(0, hx - 1), random.randint(0, hy - 1),
                random.randint(0, hz - 1), random.uniform(0, 2 * np.pi))


def RRT_Dubins_Realtime(img, start, end, stepSize, dubins_radius, plotter):
    """Run RRT with proper Dubins constraints"""
    hx, hy, hz = img.shape
    print(f"Grid shape: {hx} x {hy} x {hz}")

    dubins_planner = Dubins(radius=dubins_radius, point_separation=0.5)

    # Initialize
    node_list = [Nodes(start[0], start[1], start[2], start[3])]
    node_list[0].parent_x.append(start[0])
    node_list[0].parent_y.append(start[1])
    node_list[0].parent_z.append(start[2])
    node_list[0].parent_index = 0

    pathFound = False
    max_iterations = 10000
    i = 1
    iteration = 0

    tree_edges = []

    while not pathFound and iteration < max_iterations:
        iteration += 1

        # Goal Bias
        rand_val = random.random()
        if rand_val < 0.1:
            nx, ny, nz, n_heading = end[0], end[1], end[2], end[3]
        elif rand_val < 0.7:
            nx, ny, nz, n_heading = rnd_point(hx, hy, hz, start, end, bias_region=True)
        else:
            nx, ny, nz, n_heading = rnd_point(hx, hy, hz, bias_region=False)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Tree has {i} nodes")

        nearestIndex = nearest_node(nx, ny, nz, node_list)
        nearest_node_obj = node_list[nearestIndex]

        tx, ty, tz, t_heading, directCon, nodeCon, curve_3d = check_collision_dubins(
            nx, ny, nz, n_heading,
            nearest_node_obj.x, nearest_node_obj.y, nearest_node_obj.z, nearest_node_obj.heading,
            img, stepSize, end, dubins_planner
        )

        if nodeCon:
            new_node = Nodes(tx, ty, tz, t_heading)
            new_node.parent_index = nearestIndex
            new_node.curve_to_parent = curve_3d  # Store smooth curve
            new_node.parent_x = nearest_node_obj.parent_x.copy()
            new_node.parent_y = nearest_node_obj.parent_y.copy()
            new_node.parent_z = nearest_node_obj.parent_z.copy()
            new_node.parent_x.append(tx)
            new_node.parent_y.append(ty)
            new_node.parent_z.append(tz)

            node_list.append(new_node)

            # Draw tree edge
            if curve_3d and len(curve_3d) > 1:
                seg_array = np.array(curve_3d)
                edge_actor = vedo.Line(seg_array, c='cyan', lw=1, alpha=0.3)
                plotter.add(edge_actor)
                tree_edges.append(edge_actor)

            # Draw node
            node_actor = vedo.Points([(tx, ty, tz)], c='blue', r=2)
            plotter.add(node_actor)

            if iteration % 5 == 0:
                plotter.render()

            i += 1

            if directCon:
                print(f"\n✓ Path found after {iteration} iterations!")
                print(f"  Tree explored {len(node_list)} nodes")

                # Final connection to goal
                final_curve_2d = dubins_planner.dubins_path(
                    (tx, ty, t_heading),
                    (end[0], end[1], end[3])
                )

                z_start, z_end = tz, end[2]
                final_3d = []
                for i_pt in range(len(final_curve_2d)):
                    t_z = i_pt / (len(final_curve_2d) - 1) if len(final_curve_2d) > 1 else 0
                    z_interp = z_start + t_z * (z_end - z_start)
                    final_3d.append((final_curve_2d[i_pt][0], final_curve_2d[i_pt][1], z_interp))

                # Hide tree edges
                print("\nHiding tree visualization...")
                for edge in tree_edges:
                    plotter.remove(edge)

                # Reconstruct solution path
                print("\n=== Reconstructing Solution Path ===")
                path_segments = []
                current_idx = len(node_list) - 1

                while current_idx != 0:
                    node = node_list[current_idx]
                    if node.curve_to_parent:
                        path_segments.append(list(node.curve_to_parent))
                    else:
                        path_segments.append([(node.x, node.y, node.z)])
                    current_idx = node.parent_index

                path_segments.reverse()

                # Build continuous path
                solution_path = []
                for i, segment in enumerate(path_segments):
                    if i == 0:
                        solution_path.extend(segment)
                    else:
                        if len(solution_path) > 0 and len(segment) > 0:
                            last_pt = solution_path[-1]
                            first_seg_pt = segment[0]
                            dist = math.sqrt((last_pt[0] - first_seg_pt[0]) ** 2 +
                                             (last_pt[1] - first_seg_pt[1]) ** 2 +
                                             (last_pt[2] - first_seg_pt[2]) ** 2)
                            if dist < 0.5:
                                solution_path.extend(segment[1:])
                            else:
                                solution_path.extend(segment)
                        else:
                            solution_path.extend(segment)

                solution_path.extend(final_3d)

                # Draw solution path
                solution_array = np.array(solution_path)
                plotter.add(vedo.Tube(solution_array, c='red', r=3))
                plotter.add(vedo.Line(solution_array, c='yellow', lw=3))

                pathFound = True
                plotter.render()
                break

    if not pathFound:
        print(f"✗ Failed to find path after {max_iterations} iterations")

    return pathFound, node_list


# ------------
# MAIN
# ------------

if __name__ == '__main__':
    print("Generating 3D array...")
    grid = np.array(generate3DArray())
    generateSTL(grid)
    print(f"Grid generated: {grid.shape}")

    start = [35, 192, 242, 0.0]
    end = [45, 280, 200, np.pi / 4]
    stepSize = 8
    dubins_radius = 10.0

    print(f"\nStart: {start}")
    print(f"End: {end}")
    print(f"Step size: {stepSize}")
    print(f"Dubins turning radius: {dubins_radius}")
    print()

    # Setup visualization
    plotter = vedo.Plotter(axes=1)

    mesh = vedo.load("mesh/mesh_1.stl")
    mesh.alpha(0.15)
    plotter.add(mesh)

    plotter.add(vedo.Points([start[:3]], c='green', r=10))
    plotter.add(vedo.Points([end[:3]], c='red', r=10))

    plotter.camera.SetPosition(200, 200, 200)
    plotter.camera.SetFocalPoint(35, 220, 220)
    plotter.camera.SetViewUp(0, 0, 1)

    plotter.show(interactive=False)

    print("Running RRT with proper Dubins constraints")
    pathFound, node_list = RRT_Dubins_Realtime(grid, start, end, stepSize, dubins_radius, plotter)

    if pathFound:
        print("\n Solution Path shown in RED")
    else:
        print("Failed to find path")

    plotter.interactive()
