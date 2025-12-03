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
# DUBINS PATH PLANNER (2D) - Simplified RRT
# -----------------------------------------

def dist(pt_a, pt_b):
    return math.sqrt((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2)


def mod2pi(theta):
    """Normalize angle to [-pi, pi]"""
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta < -np.pi:
        theta += 2 * np.pi
    return theta


class Dubins:
    """2D Dubins path planner with turning radius constraint"""

    def __init__(self, radius, point_separation):
        assert radius > 0 and point_separation > 0
        self.radius = radius
        self.point_separation = point_separation

    def dubins_path(self, start, end):
        """
        Generate Dubins-inspired curved path
        Uses smooth arcs that respect turning radius
        """
        x0, y0, theta0 = start
        x1, y1, theta1 = end

        dx = x1 - x0
        dy = y1 - y0
        D = math.sqrt(dx ** 2 + dy ** 2)

        if D < 0.1:
            return np.array([[x0, y0], [x1, y1]])

        # Target angle
        theta_direct = math.atan2(dy, dx)

        # Angle differences
        alpha_start = mod2pi(theta_direct - theta0)
        alpha_end = mod2pi(theta1 - theta_direct)

        points = []

        # Calculate number of points for smooth curve
        num_points = max(5, int(D / self.point_separation))

        # Generate smooth path with turning constraints
        for i in range(num_points + 1):
            t = i / num_points

            # Smooth heading transition
            heading = theta0 + t * (mod2pi(theta1 - theta0))

            # Position with curved trajectory
            # Apply turning radius constraint through arc interpolation
            turn_factor = min(1.0, D / (2 * self.radius))

            # Cubic interpolation for smooth position
            t_cubic = 3 * t * t - 2 * t * t * t

            # Add curvature based on heading changes
            curve_offset_x = self.radius * math.sin(heading - theta0) * (1 - t) * turn_factor
            curve_offset_y = -self.radius * math.cos(heading - theta0) * (1 - t) * turn_factor

            x = x0 + t_cubic * dx + curve_offset_x * 0.3
            y = y0 + t_cubic * dy + curve_offset_y * 0.3

            points.append([x, y])

        return np.array(points)


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
        self.parent_index = None  # Track parent for tree structure


def dist_3d(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def check_collision_dubins(x1, y1, z1, heading1,
                           x2, y2, z2, heading2,
                           img, stepSize, end, dubins_planner):
    """
    Check collision along Dubins path from (x2,y2,z2) to (x1,y1,z1)
    WITH REASONABLE DUBINS CONSTRAINTS
    Returns: (tx, ty, tz, new_heading, directCon, nodeCon, curve_3d)
    """
    start_2d = (x2, y2, heading2)
    end_2d = (x1, y1, heading1)

    # Calculate Dubins path
    dubins_points = dubins_planner.dubins_path(start_2d, end_2d)

    if len(dubins_points) < 2:
        return x2, y2, z2, heading2, False, False, None

    # DUBINS CHECK: Verify heading compatibility (relaxed)
    dx = x1 - x2
    dy = y1 - y2
    distance_2d = math.sqrt(dx * dx + dy * dy)

    if distance_2d < 0.1:
        return x2, y2, z2, heading2, False, False, None

    heading_change = abs(mod2pi(heading1 - heading2))

    # Minimum distance needed 
    min_distance_needed = heading_change * dubins_planner.radius * 0.3  

    # Reject for extreme turns
    if distance_2d < min_distance_needed and heading_change > np.pi / 2:
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

    # Create 3D curve
    curve_3d = [(float(dubins_points[i][0]), float(dubins_points[i][1]), float(z_values[i]))
                for i in range(num_points)]

    tx, ty, tz = curve_3d[-1]

    # Calculate heading from path direction
    if len(dubins_points) >= 2:
        dx = dubins_points[-1][0] - dubins_points[-2][0]
        dy = dubins_points[-1][1] - dubins_points[-2][1]
        new_heading = math.atan2(dy, dx)
    else:
        new_heading = heading2

    hx, hy, hz = img.shape

    if tx < 0 or tx >= hx or ty < 0 or ty >= hy or tz < 0 or tz >= hz:
        return tx, ty, tz, new_heading, False, False, None

    obstacleThreshold = 230

    # Dense collision checking
    for i in range(len(dubins_points)):
        xi, yi, zi = int(dubins_points[i][0]), int(dubins_points[i][1]), int(z_values[i])

        if xi < 0 or xi >= hx or yi < 0 or yi >= hy or zi < 0 or zi >= hz:
            return tx, ty, tz, new_heading, False, False, None

        if img[xi, yi, zi] < obstacleThreshold:
            return tx, ty, tz, new_heading, False, False, None

    # Check between points with fine sampling
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
    """
    Generate random point with optional biasing toward start-goal region
    """
    if bias_region and start is not None and end is not None:
        # Sample within an expanded bounding box around start and goal
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
    """
    Run RRT with Dubins constraints - proper tree exploration
    """
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

    # Store edges for tree visualization
    tree_edges = []

    while not pathFound and iteration < max_iterations:
        iteration += 1

        # Goal Bias, needed for stl area
        rand_val = random.random()
        if rand_val < 0.1:
            # 10% - Sample exact goal
            nx, ny, nz, n_heading = end[0], end[1], end[2], end[3]
        elif rand_val < 0.7:
            # 60% - Sample in region between start and goal (focused exploration)
            nx, ny, nz, n_heading = rnd_point(hx, hy, hz, start, end, bias_region=True)
        else:
            # 30% - Sample anywhere (broad exploration)
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
            # Add new node
            new_node = Nodes(tx, ty, tz, t_heading)
            new_node.parent_index = nearestIndex
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

                # Draw final connection
                final_curve = dubins_planner.dubins_path(
                    (tx, ty, t_heading),
                    (end[0], end[1], end[3])
                )

                z_start, z_end = tz, end[2]
                final_3d = []
                for i_pt, pt in enumerate(final_curve):
                    t_z = i_pt / (len(final_curve) - 1) if len(final_curve) > 1 else 0
                    z_interp = z_start + t_z * (z_end - z_start)
                    final_3d.append((pt[0], pt[1], z_interp))

                # Extract and highlight solution path
                solution_path = []
                current_idx = len(node_list) - 1

                while current_idx != 0:
                    node = node_list[current_idx]
                    solution_path.append((node.x, node.y, node.z))
                    current_idx = node.parent_index

                solution_path.append((start[0], start[1], start[2]))
                solution_path.reverse()
                solution_path.extend(final_3d)

                # Draw RED solution path
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
    stepSize = 10  # Change the step size
    dubins_radius = 15.0  # Radius of the Dubins path

    print(f"\nStart: {start}")
    print(f"End: {end}")
    print(f"Step size: {stepSize}")
    print(f"Dubins turning radius: {dubins_radius}")
    print()

    # Setup visualization
    plotter = vedo.Plotter(axes=1)

    mesh = vedo.load("mesh/mesh_1.stl")
    mesh.alpha(0.15)  # Alpha slt value, CHANGE TO SEE INSIDE PATH
    plotter.add(mesh)

    plotter.add(vedo.Points([start[:3]], c='green', r=10))
    plotter.add(vedo.Points([end[:3]], c='red', r=10))

    plotter.camera.SetPosition(200, 200, 200)
    plotter.camera.SetFocalPoint(35, 220, 220)
    plotter.camera.SetViewUp(0, 0, 1)

    plotter.show(interactive=False)

    print("Running RRT with Dubins constraints")
    pathFound, node_list = RRT_Dubins_Realtime(grid, start, end, stepSize, dubins_radius, plotter)

    if pathFound:
        print("\n Solution Path shown in RED")

    else:
        print("Failed to find path")

    plotter.interactive()
