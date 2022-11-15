import cv2
import random
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', default="C:/Users/user/PycharmProjects/test1/image/image_Peppers512rgb.png")
#parser.add_argument('--path', default="C:/Users/user/PycharmProjects/test1/image/image_House256rgb.png")
params = parser.parse_args()


# gray, binary 이미지
img = cv2.imread(params.path, 0).astype(np.float32) / 255

# DFT를 이용하여 주파수 도메인으로 변환
dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
shifted = np.fft.fftshift(dft, axes=[0, 1])
magnitude = cv2.magnitude(shifted[:, :, 0], shifted[:, :, 1])
magnitude = np.log(magnitude)

plt.axis('off')
plt.imshow(magnitude, cmap='gray')
#plt.tight_layout(True)
plt.show()


# Diameter 및 Filter 종류 입력
Diameter = int(input("Diameter : "))
Filter = input("Filter(H or L) : ")

rows, cols = img.shape
centerX, centerY = round(rows/2), round(cols/2)
plt.figure(figsize=(10, 5))


# Low Pass Filter 적용
if Filter == 'L':
    LPF = np.zeros((rows, cols, 2),np.uint8)
    # 원 안쪽 통과
    LPF[centerX-Diameter:centerX+Diameter, centerY-Diameter:centerY+Diameter] = 1

    LPF_shift = shifted * LPF
    LPF_ishift = np.fft.ifftshift(LPF_shift)
    LPF_img = cv2.idft(LPF_ishift)
    LPF_img = cv2.magnitude(LPF_img[:, :, 0], LPF_img[:, :, 1])
    LPF_img = cv2.flip(LPF_img, 0)
    LPF_img = cv2.flip(LPF_img, 1)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.axis('off')
    plt.subplot(122), plt.imshow(LPF_img, cmap='gray')
    plt.title('Low Pass Filter'), plt.axis('off')
    plt.show()

# High Pass Filter 적용
elif Filter == 'H':
    HPF = np.ones((rows, cols, 2),np.uint8)
    # 원 바깥쪽 통과
    HPF[centerX - Diameter:centerX + Diameter, centerY - Diameter:centerY + Diameter] = 0

    HPF_shift = shifted * HPF
    HPF_ishift = np.fft.ifftshift(HPF_shift)
    HPF_img = cv2.idft(HPF_ishift)
    HPF_img = cv2.magnitude(HPF_img[:, :, 0], HPF_img[:, :, 1])
    HPF_img = cv2.flip(HPF_img, 0)
    HPF_img = cv2.flip(HPF_img, 1)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.axis('off')
    plt.subplot(122), plt.imshow(HPF_img, cmap='gray')
    plt.title('High Pass Filter'), plt.axis('off')
    plt.show()

else:
    print("only H or L")