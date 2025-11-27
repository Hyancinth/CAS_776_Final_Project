import cv2
import numpy as np
import math
import random
from pydicom import dcmread
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()
dicomPath = os.getenv("IMAGE_PATH")
ds = dcmread(dicomPath)
arr = ds.pixel_array
convertedArr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # gray scale
imgGray = convertedArr
imgInvert = cv2.bitwise_not(imgGray) # inverted gray scale
imgColor = cv2.cvtColor(imgInvert, cv2.COLOR_GRAY2BGR) # color image
