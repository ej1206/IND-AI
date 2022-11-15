import cv2
import numpy as np
import matplotlib.pyplot as plt

img0 = cv2.imread('/home/ej/WorkSpace/PycharmProjects/pythonProject/image/lena.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('/home/ej/WorkSpace/PycharmProjects/pythonProject/image/lena_rotate.png', cv2.IMREAD_GRAYSCALE)

detector = cv2.ORB_create(100) #  아 feature를 100개 추출
kps0, fea0 = detector.detectAndCompute(img0, None)
kps1, fea1 = detector.detectAndCompute(img1, None)

matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False) # 매쳐 생성
matches = matcher.match(fea0, fea1)                     # 매쳐로 매칭

pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1, 2)
pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1, 2)
H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0) ## RANSAC을 통해 매칭된 피쳐들간의 Homography Matrix 구함 (Treshold 작게할수록 정확한것만 매칭됨)

plt.figure()
plt.subplot(211)
plt.axis('off')
plt.title('all matches')
dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, matches, None) # mask에 있는 모든 것들 출력
plt.imshow(dbg_img[:,:,[2,1,0]])

plt.subplot(212)
plt.axis('off')
plt.title('filtered matches')
dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, [m for i, m in enumerate(matches) if mask[i]], None) # 마스크에서 맞는것만(a aocl) 출력
plt.imshow(dbg_img[:,:,[2,1,0]])
plt.tight_layout

plt.savefig("result.png")