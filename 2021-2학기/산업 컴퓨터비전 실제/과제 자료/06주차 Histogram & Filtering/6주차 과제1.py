import cv2
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', default="C:/Users/user/PycharmProjects/test1/image/lena.png")
params = parser.parse_args()

# 각 채널의 히스토그램 및 평탄화 이미지 출력
def histogram(color):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    height, width, num = img_yuv.shape
    R, G, B = cv2.split(img)

    Red = np.zeros(256, np.int32)
    Green = np.zeros(256, np.int32)
    Blue = np.zeros(256, np.int32)

    for i in range(height):
        for j in range(width):
            Red[R[i][j]] += 1
            Green[G[i][j]] += 1
            Blue[B[i][j]] += 1

    if color == 'r':
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        plt.plot(Red, color='r')

    elif color == 'g':
        img_yuv[:, :, 1] = cv2.equalizeHist(img_yuv[:, :, 1])
        plt.plot(Green, color='g')

    elif color == 'b':
        img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
        plt.plot(Blue, color='b')

    plt.show()
    img_result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow("Original img", img)
    cv2.imshow('Channel Histogram', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 원본 이미지 load
img = cv2.imread(params.path)

# 채널 선택 및 histogram 함수 호출
color = input("채널을 입력하세요 : ")
histogram(color)
