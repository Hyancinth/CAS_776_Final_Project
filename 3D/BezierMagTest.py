import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def computeTangents(waypoints):
    """
    For each waypoint, compute a tangent vector as the normalized average
    of the incoming and outgoing segment directions.
    For first/last waypoints, use forward/backward difference respectively.
    Returns list of tangent unit-vectors (numpy array per waypoint).
    """
    n = len(waypoints)
    tangents = []
    for i in range(n):
        if i == 0:
            v = np.array(waypoints[1]) - np.array(waypoints[0])
        elif i == n - 1:
            v = np.array(waypoints[-1]) - np.array(waypoints[-2])
        else:
            v1 = np.array(waypoints[i]) - np.array(waypoints[i-1])
            v2 = np.array(waypoints[i+1]) - np.array(waypoints[i])
            v = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            tangents.append(np.zeros(3))
        else:
            tangents.append(v / norm)
    return tangents

def hermite_to_bezier(p0, p1, m0, m1):
    """
    Given endpoints p0, p1 and tangent directions m0, m1 (unit vectors),
    return cubic Bézier control points [P0, P1, P2, P3] so that:
      - curve starts at p0, ends at p1
      - tangent at p0 ~ m0, tangent at p1 ~ m1
    We scale the tangents by the chord length / 3 as heuristic.
    """
    P0 = np.array(p0, dtype=float)
    P3 = np.array(p1, dtype=float)
    chord = P3 - P0
    L = np.linalg.norm(chord)
    # handle degenerate case
    if L < 1e-6:
        return [P0, P0, P0, P0]
    P1 = P0 + m0 * (L / 3.0)
    P2 = P3 - m1 * (L / 3.0)
    return [P0, P1, P2, P3]

def eval_cubic_bezier(ctrl_pts, num_points=100):
    P0, P1, P2, P3 = ctrl_pts
    t = np.linspace(0, 1, num_points)
    u = 1 - t
    B = (u**3)[:, None]*P0 + 3*(u**2 * t)[:, None]*P1 + \
        3*(u * t**2)[:, None]*P2 + (t**3)[:, None]*P3
    return B

def build_beziers_from_waypoints(waypoints):
    tangents = computeTangents(waypoints)
    beziers = []
    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i+1]
        m0 = tangents[i]
        m1 = tangents[i+1]
        bez = hermite_to_bezier(p0, p1, m0, m1)
        beziers.append(bez)
    return beziers

def plot_curve(beziers, waypoints):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ws = np.array(waypoints)
    ax.scatter(ws[:,0], ws[:,1], ws[:,2], c='red', marker='o', s=50, label='Waypoints')

    # Plot each Bézier segment
    for bez in beziers:
        pts = eval_cubic_bezier(bez, num_points=200)
        ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=2)

        # Also draw the control polygon (handles)
        cps = np.array(bez)
        ax.plot(cps[:,0], cps[:,1], cps[:,2], linestyle='--', color='gray')
        ax.scatter(cps[:,0], cps[:,1], cps[:,2], c='blue', marker='^', s=30)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Cubic Bezier through waypoints, tangent = avg direction')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Example waypoints — replace with your RRT output
    waypoints = [
        (0.0, 0.0, 0.0),
        (30.0, 5.0,  0.0),
        (60.0, 20.0, 10.0),
        (100.0, 50.0, 5.0),
        (130.0, 80.0, 0.0)
    ]

    beziers = build_beziers_from_waypoints(waypoints)
    plot_curve(beziers, waypoints)
