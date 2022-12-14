import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/ej/WorkSpace/PycharmProjects/pythonProject/image/lena.png', cv2.IMREAD_GRAYSCALE)
corners = cv2.goodFeaturesToTrack(img, 100, 0.05, 10)

for c in corners:
    x, y = c[0]
    cv2.circle(img, (x, y), 5, 255, -1)

plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.savefig("feature_lena.png")