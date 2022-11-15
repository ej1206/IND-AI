import cv2
import random
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', default="C:/Users/user/PycharmProjects/test1/image/image_Peppers512rgb.png")
params = parser.parse_args()

# gray, binary 이미지
gray = cv2.imread(params.path, 0)
ret, img = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

# Erosion
Erosion = int(input("Erosion : "))
erosion_img = cv2.morphologyEx(img, cv2.MORPH_ERODE, (3, 3), iterations=10)

# Dilation
Dilation = int(input("Dilation : "))
dilation_img = cv2.morphologyEx(img, cv2.MORPH_DILATE, (3, 3), iterations=10)

# Opening
Opening = int(input("Opening : "))
opening_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=Opening)

# Closing
Closing = int(input("Closing : "))
closing_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=Closing)

# Result 출력
plt.figure(figsize=(10, 8))
plt.subplot(231), plt.axis('off')
plt.title("Original"), plt.imshow(img, cmap='gray')

plt.subplot(232), plt.axis('off')
plt.title("Erosion"), plt.imshow(erosion_img, cmap='gray')

plt.subplot(233), plt.axis('off')
plt.title("Dilation"), plt.imshow(dilation_img, cmap='gray')

plt.subplot(234), plt.axis('off')
plt.title("Opening"), plt.imshow(opening_img, cmap='gray')

plt.subplot(235), plt.axis('off')
plt.title("Closing"), plt.imshow(closing_img, cmap='gray')

plt.show()