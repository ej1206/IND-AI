import cv2
import numpy as np

img = cv2.imread('/home/ej/WorkSpace/PycharmProjects/pythonProject/image/scenetext.jpg', cv2.IMREAD_COLOR)
corners = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)

corners = cv2.dilate(corners, None)

show_img = np.copy(img)
show_img[corners>0.1*corners.max()]=[0, 0, 255] # 에.. 빨간점... 

corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show_img = np.hstack((show_img, cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR))) # 두번째 사진은 nomalize한것.
                                                                            # 코너를 0~255까지

cv2.imshow('Harris corner detector', show_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

fast = cv2.FastFeatureDetector_create(30, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16) # 16개의 점중에 9개의 점이 크거나 작으면 코너다.
kp = fast.detect(img)

show_img = np.copy(img)
for p in cv2.KeyPoint_convert(kp):
    cv2.circle(show_img, tuple(p), 2, (0, 255, 0), cv2.FILLED)

cv2. imshow('Fast corner detector', show_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

fast.setNonmaxSuppression(False) # NonmaxSuppression 안함
kp = fast.detect(img)

for p in cv2.KeyPoint_convert(kp):
    cv2.circle(show_img, tuple(p), 2, (0, 255, 0), cv2.FILLED)

cv2. imshow('Fast corner detector', show_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()