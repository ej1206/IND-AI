# Estimating disparity map for stereo images
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 20})

left_img = cv2.imread('C:/Users/user/Desktop/right.png')
right_img = cv2.imread('C:/Users/user/Desktop/left.png')


cv2.imshow("left", left_img)
cv2.imshow("right", right_img)

cv2.waitKey()
cv2.destroyAllWindows()

stereo_bm = cv2.StereoBM_create(32)
dispmap_bm = stereo_bm.compute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))


stereo_sgbm = cv2.StereoSGBM_create(0, 32)
dispmap_sgbm = stereo_sgbm.compute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))


plt.figure(0, figsize=(12, 10))
plt.subplot(221)
plt.title('left')
plt.imshow(left_img[:, :, [2, 1, 0]])

plt.subplot(222)
plt.title('right')
plt.imshow(right_img[:, :, [2, 1, 0]])

plt.subplot(223)
plt.title('BM')
plt.imshow(dispmap_bm, cmap='gray')

plt.subplot(224)
plt.title('SGBM')
plt.imshow(dispmap_sgbm, cmap='gray')
plt.show()



