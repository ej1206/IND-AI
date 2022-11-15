# Resize, Flipping

import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='C:/Users/user/Desktop/lena.png', help='Image Path')
params = parser.parse_args()
img = cv2.imread(params.path)
print('original img.shape: ', img.shape)
cv2.imshow("original", img)
cv2.waitKey(0)


# Resize
width, height = 128, 256
resizedImg = cv2.resize(img, (width, height))
print('resized to 128x256 image.shape: ', resizedImg.shape)
cv2.imshow("resized1", resizedImg)
cv2.waitKey(0)


# Resize 가로 * 0.25, 세로 * 0.5
w_mult, h_mult = 0.25, 0.5
resizedImg = cv2.resize(img, (0, 0), resizedImg, w_mult, h_mult)
print('img.shape: ', resizedImg.shape)
cv2.imshow("resized2", resizedImg)
cv2.waitKey(0)


# Resize 가로 * 2, 세로 * 4
w_mult, h_mult = 2, 4
resizedImg = cv2.resize(img, (0, 0), resizedImg, w_mult, h_mult, cv2.INTER_NEAREST)
print('img.shape: ', resizedImg.shape)
cv2.imshow("resized3", resizedImg)
cv2.waitKey(0)


# Flip
img_flip_along_x = cv2.flip(img, 0)
cv2.imshow("flipped1", img_flip_along_x)
cv2.waitKey(0)

# Flip
img_flip_along_x_along_y = cv2.flip(img_flip_along_x, 1)
cv2.imshow("flipped2", img_flip_along_x_along_y)
cv2.waitKey(0)

# Flip
img_flipped_xy = cv2.flip(img, -1)
cv2.imshow("flipped3", img_flipped_xy)
cv2.waitKey(0)

# Check That Sequential Filps Around x and y Equal to Simultaneous x-y filp
assert img_flipped_xy.all() == img_flip_along_x_along_y.all()