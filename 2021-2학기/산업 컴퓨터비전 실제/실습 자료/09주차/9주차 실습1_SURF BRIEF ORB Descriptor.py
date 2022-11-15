import cv2

img = cv2.imread('/home/ej/WorkSpace/PycharmProjects/pythonProject/image/scenetext.jpg', cv2.IMREAD_COLOR)

surf = cv2.xfeatures2d.SURF_create(10000) 
surf.setExtended(True)  # extended 할지말지
surf.setNOctaves(3)     # octave 개수
surf.setNOctaveLayers(10)
surf.setUpright(False)

keyPoints, descriptors = surf.detectAndCompute(img, None) # surf로 디스크립터, 키포인트 추출

show_img = cv2.drawKeypoints(img, keyPoints, None, (255, 0, 0),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SURF descriptors', show_img)
cv2.waitKey()
cv2.destroyAllWindows()

brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(32,True)

KeyPoints, descriptors = brief.compute(img, keyPoints)

show_img = cv2.drawKeypoints(img, keyPoints, None, (0, 255, 0),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('BRIEF descriptors', show_img)
cv2.waitKey()
cv2.destroyAllWindows()

orb = cv2.ORB_create()
orb.setMaxFeatures(200)

keyPoints = orb.detect(img, None)
keyPoints, descriptors = orb.compute(img, keyPoints)

show_img = cv2.drawKeypoints(img, keyPoints, None, (0, 0, 255),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('ORB descriptors', show_img)
cv2.waitKey()
cv2.destroyAllWindows()