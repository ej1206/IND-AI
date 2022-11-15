# Connected Componet
# 바이너리 영상에 대해서만 적용 가능

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/user/PycharmProjects/test1/a.png", cv2.IMREAD_GRAYSCALE)

connectivity = 8됨
num_labels, labelmap = cv2.connectedComponents(img, connectivity, cv2.CV_32S)         # 이미지, connectivity  넣어서 레이블의 수, 각각의 레이블
img = np.hstack((img, labelmap.astype(np.float32)/(num_labels - 1)))                  # 이미지를 스태킹 해서 영역 구분
cv2.imshow('connected components', img)
cv2.waitKey()
cv2.destroyAllWindows()

img = cv2.imread("C:/Users/user/PycharmProjects/test1/lena.png", cv2.IMREAD_GRAYSCALE)
otsu_thr, otsu_mask = cv2.threshold(img, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 이진화

output = cv2.connectedComponentsWithStats(otsu_mask, connectivity, cv2.CV_32S)

num_labels, labelmap, stats, centers = output

colored = np.full((img.shape[0], img.shape[1], 3), 0, np.uint8) #각각의 label마다 다른 색깔


for l in range(1, num_labels):
    if stats[l][4] > 200: # 같은 레이블을 200개 이상 가지면
        colored[labelmap == l] = (0, 255*l/num_labels, 255*(num_labels-l)/num_labels)             # l마다 랜덤컬러 적용
        cv2.circle(colored, (int(centers[l][0]), int(centers[l][1])), 5, (255, 0, 0), cv2.FILLED) # 센터에다가 파란색 점 그리기

img = cv2.cvtColor(otsu_mask*255, cv2.COLOR_GRAY2BGR)
cv2.imshow('connected components', np.hstack((img, colored)))
cv2.waitKey()
cv2.destroyAllWindows()