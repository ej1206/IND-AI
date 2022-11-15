import cv2
import numpy as np
import os

camera_matrix = np.load('camera_mat.npy')
dist_coefs = np.load('dist_coefs.npy')

img = cv2.imread('C:/Users/user/Desktop/camera_cali/img_00.png')
pattern_size = (10, 7)
res, corners = cv2.findChessboardCorners(img, pattern_size)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners, (10, 10), (-1, -1), criteria)

h_corners = cv2.undistortPoints(corners, camera_matrix, dist_coefs)
h_corners = np.c_[h_corners.squeeze(), np.ones(len(h_corners))]

img_pts, _ = cv2.projectPoints(h_corners, (0, 0, 0), (0, 0, 0), camera_matrix, None)

for c in img_pts.squeeze().astype(np.float32):
    cv2.circle(img, (int(c[0]), int(c[1])), 5, (0, 0, 255), 2)

cv2.imshow('undistorted corners', img)
cv2.waitKey()
cv2.destroyAllWindows

img_pts, _ = cv2.projectPoints(h_corners, (0, 0, 0), (0, 0, 0), camera_matrix, dist_coefs)

for c in img_pts.squeeze().astype(np.float32):
    cv2.circle(img, (int(c[0]), int(c[1])), 2, (255, 255, 0), 2)

cv2.imshow('reprojected corners', img)
cv2.waitKey()
cv2.destroyAllWindows()

# 흠.. 빨간색이 왜곡 보상된 점 녹색이 원래 점
# 가운데로 갈수록 빨간색과 녹색이 거의 일치, 외곽으로 갈수록 왜곡이 크다.
# 녹색의 중심이 파란색이다 = 캘리가 잘됐다