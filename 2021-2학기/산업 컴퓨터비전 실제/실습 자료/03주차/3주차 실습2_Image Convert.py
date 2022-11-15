import cv2
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default="/home/ej/Downloads/Lenna.png")
parser.add_argument('--outputPath', default="/home/ej/Downloads/Lenna_Result.png")
params = parser.parse_args()


img = cv2.imread(params.path)
print('Shape : ', img.shape)
print('Data Type : ', img.dtype)
cv2.imshow("Original Image", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Converted to Grayscale")
print('Shape : ', gray.shape)
print('Data Type : ', gray.dtype)
cv2.imshow("Gray-Scale Image", gray)

hsv = gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print("Converted to HSV")
print('Shape : ', hsv.shape)
print('Data Type : ', hsv.dtype)

hsv[:, :, 2] *= 2
from_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
print("Converted back to BGR from HSV")
print('Shape : ', from_hsv.shape)
print('Data Type : ', from_hsv.dtype)
cv2.imshow("from_hsv", from_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()