import cv2
import numpy as np

# Panorama 생성 함수
def Make_Panorama(images):
    stitcher = cv2.createStitcher()
    ret, panorama = stitcher.stitch(images)

    if ret == cv2.STITCHER_OK:
        panorama = cv2.resize(panorama, dsize=(0, 0), fx=0.2, fy=0.2)
    else:
        print('Error during stitching')

    cv2.imshow("panorama", panorama)
    cv2.imwrite('/home/ej/Desktop/stitching/result.jpg', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# boat Image Load
boat = []
for i in range(1, 7, 1):
    imgName = '/home/ej/Desktop/stitching/boat{0}.jpg'.format(i)
    boat.append(cv2.imread(imgName, cv2.IMREAD_COLOR))

# budapest Image Load
budapest = []
for i in range(1, 7, 1):
    imgName = '/home/ej/Desktop/stitching/budapest{0}.jpg'.format(i)
    budapest.append(cv2.imread(imgName, cv2.IMREAD_COLOR))


# newspaper Image Load
newspaper = []
for i in range(1, 5, 1):
    imgName = '/home/ej/Desktop/stitching/newspaper{0}.jpg'.format(i)
    newspaper.append(cv2.imread(imgName, cv2.IMREAD_COLOR))

# s Image Load
s = []
s.append(cv2.imread('/home/ej/Desktop/stitching/s1.jpg', cv2.IMREAD_COLOR))
s.append(cv2.imread('/home/ej/Desktop/stitching/s2.jpg', cv2.IMREAD_COLOR))

#Make_Panorama(boat)
#Make_Panorama(budapest)
#Make_Panorama(newspaper)
Make_Panorama(s)
