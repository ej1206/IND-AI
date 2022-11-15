import cv2
import numpy as np
import matplotlib.pyplot as plt

color = np.random.randint(0,255,(100,3))


def Optical_Flow(img1, img2, gray_img1, gray_img2, corner1):
    mask = np.zeros_like(img1)

    # Calculate Optical Flow using Pytamid Lucas-Kanade
    corner2, status, errors = cv2.calcOpticalFlowPyrLK(
        gray_img1, gray_img2, corner1, None, winSize=(15, 15), maxLevel=5,  # 피라미드를 얼마나 쌓을지
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Select Good Points
    good_new = corner2[status == 1]
    good_old = corner1[status == 1]

    # Draw Optical Flow
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 255), 2)
        img2 = cv2.circle(img2, (a, b), 5, (255, 0, 0), -1)


    img_show = cv2.add(img2, mask)
    cv2.imshow('frame', img_show)
    k = cv2.waitKey(0)
    if k == 27:
        exit(0)

def goodFeaturesToTrack(img1, img2):

    # Create Gray Image
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Extract Each Image's Corner Using goodFeaturesToTrack
    corner1 = cv2.goodFeaturesToTrack(gray_img1, 100, 0.05, 10)
    corner2 = cv2.goodFeaturesToTrack(gray_img2, 100, 0.05, 10)

    # Copy Image
    img1_show = img1.copy()
    img2_show = img2.copy()

    # Draw Circle on Each Image
    for c in corner1:
        x, y = c[0]
        cv2.circle(img1_show, (x, y), 5, (0, 255, 0), -1)

    for c in corner2:
        x, y = c[0]
        cv2.circle(img2_show, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('corner1', img1_show)
    cv2.imshow('corner2', img2_show)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Function 2 Optical_Flow
    Optical_Flow(img1, img2, gray_img1, gray_img2, corner1)


# Image Load, Resize
img1 = cv2.imread('/home/ej/Desktop/stitching/dog_a.jpg', cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, dsize=(0, 0), fx=0.5, fy=0.5)
img2 = cv2.imread('/home/ej/Desktop/stitching/dog_b.jpg', cv2.IMREAD_COLOR)
img2 = cv2.resize(img2, dsize=(0, 0), fx=0.5, fy=0.5)

# Function 1 goodFeaturesToTrack
goodFeaturesToTrack(img1, img2)


