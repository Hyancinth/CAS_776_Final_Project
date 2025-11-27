import numpy as np
from pydicom import dcmread
import cv2
from dotenv import load_dotenv
import os
from skimage import measure
from stl import mesh

def generate3DArray():
    load_dotenv()
    invertedImgs = []
    for i in range(125, 300):
        filename = f"{os.getenv('FOLDER_PATH')}/0500M-01 PJW Sinus Model - Nose {40000 + i}.dcm"
        ds = dcmread(filename)
        print("Reading file:", filename)
        arr = ds.pixel_array
        convertedArr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        imgGray = convertedArr
        imgInvert = cv2.bitwise_not(imgGray)
        imgColor = cv2.cvtColor(imgInvert, cv2.COLOR_GRAY2BGR)
        invertedImgs.append(imgColor)

    stackedData = np.stack(invertedImgs, axis=0)
    print("Stacked data shape:", stackedData.shape)

    data3D = stackedData[:,:,:,0] 
    print("3D data shape:", data3D.shape)

    return data3D

def generateSTL(data3D):
    verts, faces, normals, values = measure.marching_cubes(data3D, level=120, step_size=5)
    # increase/decrease step_size for more/less detail

    obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        obj_3d.vectors[i] = verts[f]

    os.makedirs('mesh', exist_ok=True)
    file_path = 'mesh/mesh_1.stl'
    obj_3d.save(file_path)

    print(f"Successfully generated STL file: {file_path}")

    return file_path

if __name__ == '__main__':
    data3D = generate3DArray()
    stl_file = generateSTL(data3D)