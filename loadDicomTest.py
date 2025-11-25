from pydicom import dcmread
import matplotlib.pyplot as plt
import numpy as np
import cv2

path = "0500M-01 PJW Sinus Model - Nose 4 (DICOM)/0500M-01 PJW Sinus Model - Nose 40162.dcm"
ds = dcmread(path)

arr = ds.pixel_array
print(arr)

# cv2.imshow("Dicom Image", arr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# convertedArr = (arr*255).round().astype(np.uint8)
convertedArr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# arr = cv2.applyColorMap(convertedArr, cv2.COLORMAP_JET)
cv2.imshow("Dicom Image", convertedArr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(ds.pixel_array)
# plt.show()