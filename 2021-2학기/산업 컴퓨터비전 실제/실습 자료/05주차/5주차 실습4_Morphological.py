# 4) Morphological Filter 
# Erosion 10 Dialation, closing, opening 이런거 

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("C:/Users/user/PycharmProjects/test1/lena.png", 0) #흑백으로

thr, mask = cv2.threshold(image, 200, 1, cv2.THRESH_BINARY)           #threshold
print('threshold used:', thr)

adapt_mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 10)
#어떤 #threshold 를 해도 정확히 나올 순 없음 이렇게 Adaptive 하는게 더 조음 Otu's threshold랑 adaptive threshold랑 비교
#Otu는  threshold 하나만 구해도 저렇게 잘나옴

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap='gray')

plt.subplot(132)
plt.axis('off')
plt.title('binary threshold')
plt.imshow(mask, cmap='gray')

plt.subplot(133)
plt.axis('off')
plt.title('adaptive threshold')
plt.imshow(adapt_mask, cmap='gray')
plt.tight_layout
plt.show()