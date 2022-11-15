import cv2
import random
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', default="C:/Users/user/PycharmProjects/test1/image/lena.png")
params = parser.parse_args()


# 1) original, gray, noise image 로드 및 생성
img = cv2.imread(params.path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_noise = np.zeros(img.shape, dtype=img.dtype)
width = img.shape[0]
height = img.shape[1]
channel = img.shape[2]

# 2) 랜덤 노이즈 생성
for i in range(width):
    for j in range(height):
        rand = random.randrange(-30, 30)
        img_noise[i, j] = img_gray[i, j] + rand

# 3) diameter, SigmaColor, SigmaSpace 입력
diameter = int(input("diameter : "))
SigmaColor = int(input("SigmaColor : "))
SigmaSpace = int(input("SigmaSpace : "))

# 4) original, noise, noise 제거 image 출력
img_result = cv2.bilateralFilter(img_noise, diameter, sigmaColor=SigmaColor, sigmaSpace=SigmaSpace)
cv2.imshow("original img", img_gray)
cv2.imshow("noise img", img_noise)
cv2.imshow("bilateralFilter", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()


