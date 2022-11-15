# 2. Image sharpening using Unsharp mask
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

image = cv2.imread("C:/Users/user/PycharmProjects/test1/lena.png")

KSIZE = 11
ALPHA = 2 # 이거 값 두개 바꿔가면서 
Kernel = cv2.getGaussianKernel(KSIZE, 0)
Kernel = -ALPHA * Kernel @ Kernel.T
Kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA

print(Kernel.shape, Kernel.dtype, Kernel.sum())

filtered = cv2.filter2D(image, -1, Kernel)


plt.figure(figsize=(8,4))
plt.subplot(121)
plt.axis('off')
plt.title('image')
plt.imshow(image[:, :, [2, 1, 0]])


plt.subplot(122)
plt.axis('off')
plt.title('filtered')
plt.imshow(image[:, :, [2, 1, 0]])
plt.tight_layout(True)
plt.show()

cv2.imshow('before', image)
cv2.imshow('after', filtered)
cv2.waitKey()
cv2.destroyAllWindows()