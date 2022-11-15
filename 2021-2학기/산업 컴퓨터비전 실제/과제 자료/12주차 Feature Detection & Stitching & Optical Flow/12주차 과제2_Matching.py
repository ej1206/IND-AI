import cv2
import numpy as np
import matplotlib.pyplot as plt

def ORB_Warping(img1, img2):

    # ORB Detector
    ORB = cv2.ORB_create()
    ORB.setMaxFeatures(200)

    # Keypoint, Feature
    kps1, des1 = ORB.detectAndCompute(img1, None)
    kps2, des2 = ORB.detectAndCompute(img2, None)

    # Draw Keypoint
    DrawKeypoint1 = cv2.drawKeypoints(img1, kps1, None,(0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    DrawKeypoint2 = cv2.drawKeypoints(img2, kps2, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Match Keypoint
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)  # 매쳐 생성
    matches = matcher.match(des1, des2)  # 매쳐로 매칭
    s_match = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # Find Homography using RANSAC
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3,
                                 0)

    img = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))

    cv2.imshow('ORB Detector KeyPoints', DrawKeypoint1)
    cv2.imshow('ORB Detector  KeyPoints', DrawKeypoint2)
    cv2.imshow('ORB Detector  match', s_match)
    cv2.imshow('img warping', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def SURF_Warping(img1, img2):

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SURF Detector
    SURF = cv2.xfeatures2d.SURF_create(1000)
    SURF.setExtended(True)  # extended 할지말지
    SURF.setNOctaves(3)  # octave 개수
    SURF.setNOctaveLayers(10)
    SURF.setUpright(False)

    # Keypoint, Feature
    kps1, des1 = SURF.detectAndCompute(img1, None)
    kps2, des2 = SURF.detectAndCompute(img2, None)

    # Draw Keypoint
    DrawKeypoint1 = cv2.drawKeypoints(img1, kps1, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    DrawKeypoint2 = cv2.drawKeypoints(img2, kps2, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    # matcher 생성
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), )
    matches = matcher.match(des1, des2)  # 매쳐로 매칭
    s_match = cv2.drawMatches(img1, kps1, img2, kps2, matches, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # Find Homography using RANSAC
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3,
                                 0)

    img = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))

    cv2.imshow('SURF Detector  KeyPoints', DrawKeypoint1)
    cv2.imshow('SURF Detector KeyPoints', DrawKeypoint2)
    cv2.imshow('SURF Detector match', s_match)
    cv2.imshow('img warping', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def SIFT_Warping(img1, img2):

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT Detector
    SIFT = cv2.xfeatures2d.SIFT_create()

    # Keypoint, Feature
    kps1, des1 = SIFT.detectAndCompute(img1, None)
    kps2, des2 = SIFT.detectAndCompute(img2, None)

    # Draw Keypoint
    DrawKeypoint1 = cv2.drawKeypoints(img1, kps1, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    DrawKeypoint2 = cv2.drawKeypoints(img2, kps2, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    # matcher 생성
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), )
    matches = matcher.match(des1, des2)  # 매쳐로 매칭
    s_match = cv2.drawMatches(img1, kps1, img2, kps2, matches, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # Find Homography using RANSAC
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3, 0)

    img = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))

    cv2.imshow('SIFT Detector KeyPoints', DrawKeypoint1)
    cv2.imshow('SIFT KeyPoints', DrawKeypoint2)
    cv2.imshow('SIFT match', s_match)
    cv2.imshow('img warping', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# boat Image Load
img1 = cv2.imread('/home/ej/Desktop/stitching/boat1.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('/home/ej/Desktop/stitching/boat2.jpg', cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
img2 = cv2.resize(img2, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
ORB_Warping(img1, img2)
SURF_Warping(img1, img2)
SIFT_Warping(img1, img2)


# budapest Image Load
img1 = cv2.imread('/home/ej/Desktop/stitching/budapest1.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('/home/ej/Desktop/stitching/budapest2.jpg', cv2.IMREAD_COLOR)
ORB_Warping(img1, img2) # 이상함
SURF_Warping(img1, img2)
SIFT_Warping(img1, img2)


# s Image Load
img1 = cv2.imread('/home/ej/Desktop/stitching/s1.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('/home/ej/Desktop/stitching/s2.jpg', cv2.IMREAD_COLOR)
ORB_Warping(img1, img2)
SURF_Warping(img1, img2)
SIFT_Warping(img1, img2)

