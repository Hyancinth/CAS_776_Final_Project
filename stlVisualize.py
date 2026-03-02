import vedo
from vedo import load, show

mesh = load("mesh/mesh_1.stl")
mesh.alpha(0.1) # transparency
slicedMesh = mesh.cut_with_plane(origin = (0, 0, 256), normal="z")
show(slicedMesh)