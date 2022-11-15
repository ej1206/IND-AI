import cv2
import numpy as np
import os

camera_matrix = np.load('camera_mat.npy')
dist_coefs = np.load('dist_coefs.npy')

img = cv2.imread('C:/Users/user/Desktop/camera_cali/img_00.png')
cv2.imshow('original image', img)

ud_img = cv2.undistort(img, camera_matrix, dist_coefs) # 이미지를 입력받아서 undistorted 이미지 생성
cv2.imshow('undistorted image1', ud_img)

opt_cam_mat, valid_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, img.shape[:2][::-1], 0) # optimal하게 undistort img 생성
ud_img = cv2.undistort(img, camera_matrix, dist_coefs, None, opt_cam_mat)
cv2.imshow('undistorted image2', ud_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
