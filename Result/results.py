from rrt3DBezier import RRT_Bezier
from rrt3D import RRT
from dubins3D import RRT_Dubins
from generate3D import generate3DArray
from updatedDubins3D import RRT_Dubins_Realtime
import numpy as np

class Nodes:
    """Class to store the RRT graph"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent_x = []
        self.parent_y = []
        self.parent_z = []

if __name__ == "__main__":
    stepSize = 10
    print("Generating 3D occupancy grid from DICOM files...")
    grid = np.array(generate3DArray())
    start = [35, 192, 242]
    end = [45, 280, 200]
    
    startDubins = [35, 192, 242, np.pi/4]
    endDubins = [45, 280, 200, np.pi/4]

    print("Starting RRT, RRT Bezier, and RRT Dubins path planning...")
    print("RRT")
    _, _, totalDistanceRRT, totalTimeRRT, iterationRRT, nodeCountRRT = RRT(grid, start, end, stepSize)

    print("RRT Bezier")
    _, _, totalDistanceBezier, totalTimeBezier, iterationBezier, nodeCountBezier = RRT_Bezier(grid, start, end, stepSize, bezier=True)

    print("RRT Dubins")
    _, _, totalDistanceDubins, totalTimeDubins = RRT_Dubins_Realtime(grid, startDubins, endDubins, stepSize, dubins_radius=12)

    print(f"RRT Total Distance: {totalDistanceRRT}, Time Taken: {totalTimeRRT} seconds, Iterations: {iterationRRT}, Nodes: {nodeCountRRT}")
    print(f"RRT Bezier Total Distance: {totalDistanceBezier}, Time Taken: {totalTimeBezier} seconds, Iterations: {iterationBezier}, Nodes: {nodeCountBezier}")
    print(f"RRT Dubins Total Distance: {totalDistanceDubins}, Time Taken: {totalTimeDubins} seconds")

