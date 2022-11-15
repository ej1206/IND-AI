# Aruco pattern detector
import cv2
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

img = np.full((700, 700), 255, np.uint8)

# 각 위치에 아루코 마커 그려주기
img[100:300, 100:300] = aruco.drawMarker(aruco_dict, 2, 200)
img[100:300, 400:600] = aruco.drawMarker(aruco_dict, 76, 200)
img[400:600, 100:300] = aruco.drawMarker(aruco_dict, 42, 200)
img[400:600, 400:600] = aruco.drawMarker(aruco_dict, 123, 200)

# 어렵게 가우시안도 넣어보기
img = cv2.GaussianBlur(img, (11, 11), 0)


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# 검출해보자
corners, ids, _ = aruco.detectMarkers(img, aruco_dict)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
aruco.drawDetectedMarkers(img_color, corners, ids)

cv2.imshow('Created Aruco markers', img)
cv2.imshow('Detected Aruco markers', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()