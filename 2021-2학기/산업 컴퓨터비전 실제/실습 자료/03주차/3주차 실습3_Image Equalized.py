import cv2
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default="lena.png")
parser.add_argument('--outputPath', default="/home/ej/Downloads/Lenna_Result.png")
params = parser.parse_args()


img = cv2.imread(params.path)
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original grey", grey)
cv2.waitKey(0)

hist, bins = np.histogram(grey, 256, [0, 255])
plt.fill(hist)
plt.xlabel('pixel value')
plt.show()

grey_eq = cv2.equalizeHist(grey)
hist, bins = np.histogram(grey_eq, 256, [0, 255])
plt.fill_between(range(256), hist, 0)
plt.xlabel('pixel value')
plt.show()

cv2.imshow("equlized grey", grey_eq)
cv2.waitKey(0)

color = cv2.imread(params.path)
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
color_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("original color", color)
cv2.imshow('equalized color', color_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()