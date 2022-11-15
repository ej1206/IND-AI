import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image Load
boat = cv2.imread('C:/Users/user/PycharmProjects/test1/image/stitching/boat1.jpg', cv2.IMREAD_COLOR)
budapest = cv2.imread('C:/Users/user/PycharmProjects/test1/image/stitching/budapest1.jpg', cv2.IMREAD_COLOR)
newspaper = cv2.imread('C:/Users/user/PycharmProjects/test1/image/stitching/newspaper1.jpg', cv2.IMREAD_COLOR)
s = cv2.imread('C:/Users/user/PycharmProjects/test1/image/stitching/s1.jpg', cv2.IMREAD_COLOR)

# Canny Edge
boat_canny = cv2.Canny(cv2.cvtColor(boat, cv2.COLOR_BGR2GRAY), 100, 200)
budapest_canny = cv2.Canny(cv2.cvtColor(budapest, cv2.COLOR_BGR2GRAY), 100, 200)
newspaper_canny = cv2.Canny(cv2.cvtColor(newspaper, cv2.COLOR_BGR2GRAY), 100, 200)
s_canny = cv2.Canny(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), 100, 200)

plt.figure(figsize=(20, 5))
plt.subplot(141), plt.imshow(boat_canny, cmap='gray'), plt.axis('off'), plt.title('Boat Canny Edge')
plt.subplot(142), plt.imshow(budapest_canny, cmap='gray'), plt.axis('off'), plt.title('Budapest Canny Edge')
plt.subplot(143), plt.imshow(newspaper_canny, cmap='gray'), plt.axis('off'), plt.title('Newspaper Canny Edge')
plt.subplot(144), plt.imshow(s_canny, cmap='gray'), plt.axis('off'), plt.title('S Canny Edge')
plt.show()

# Harris Corner
boat_corners = cv2.cornerHarris(cv2.cvtColor(boat, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
boat_corners = cv2.dilate(boat_corners, None)
boat[boat_corners>0.1*boat_corners.max()]=[0, 0, 255]
boat_corners = cv2.normalize(boat_corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
boat = cv2.resize(boat, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR) # resize

budapest_corners = cv2.cornerHarris(cv2.cvtColor(budapest, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
budapest_corners = cv2.dilate(budapest_corners, None)
budapest[budapest_corners>0.1*budapest_corners.max()]=[0, 0, 255]
budapest_corners = cv2.normalize(budapest_corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

newspaper_corners = cv2.cornerHarris(cv2.cvtColor(newspaper, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
newspaper_corners = cv2.dilate(newspaper_corners, None)
newspaper[newspaper_corners>0.1*newspaper_corners.max()]=[0, 0, 255]
newspaper_corners = cv2.normalize(newspaper_corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

s_corners = cv2.cornerHarris(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
s_corners = cv2.dilate(s_corners, None)
s[s_corners>0.1*s_corners.max()]=[0, 0, 255]
s_corners = cv2.normalize(s_corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


cv2.imshow('Boat Harris corner', boat)
cv2.imshow('BudapestHarris corner', budapest)
cv2.imshow('Newspaper Harris corner', newspaper)
cv2.imshow('S Harris corner', s)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()