from pydicom import dcmread
import matplotlib.pyplot as plt
import numpy as np

path = "0500M-01 PJW Sinus Model - Nose 4 (DICOM)/0500M-01 PJW Sinus Model - Nose 40055.dcm"
ds = dcmread(path)

arr = ds.pixel_array
print(arr.shape)

plt.imshow(ds.pixel_array)
plt.show()