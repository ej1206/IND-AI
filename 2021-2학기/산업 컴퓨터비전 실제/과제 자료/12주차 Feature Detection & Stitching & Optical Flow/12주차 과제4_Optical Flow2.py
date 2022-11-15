import cv2
import numpy as np


def Display_Flow(img, flow, stride=40):
    for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i * stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10 * delta)

        if 2 <= cv2.norm(delta) <= 10:
            cv2.arrowedLine(img, pt1[::-1], pt2[::-1],
                            (0, 0, 255), 5, cv2.LINE_AA, 0, 0.4)

        norm_opt_flow = np.linalg.norm(flow, axis=2)
        norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1, cv2.NORM_MINMAX)

    #cv2.imwrite('/home/ej/Desktop/stitching/optical_flowww.jpg', img)
    #cv2.imwrite('/home/ej/Desktop/stitching/optical_flowww_magnitude.jpg', norm_opt_flow)

    cv2.imshow('optical flow', img)
    cv2.imshow('optical flow magnitude', norm_opt_flow)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


def Optical_Flow_Farneback(img1, img2):

    # Create Gray Image
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    opt_flow = cv2.calcOpticalFlowFarneback(
        gray_img1, gray_img2, None, 0.5, 5, 13, 10,
        5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    Display_Flow(img2, opt_flow)


def Optical_Flow_DualTVL1(img1, img2):

    # Create Gray Image
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    Flow_DualTVL1 = cv2.createOptFlow_DualTVL1()

    if not Flow_DualTVL1.getUseInitialFlow():
        opt_flow = Flow_DualTVL1.calc(gray_img1, gray_img2, None)
        Flow_DualTVL1.setUseInitialFlow(True)

    Display_Flow(img2, opt_flow)


# Image Load, Resize
img1 = cv2.imread('/home/ej/Desktop/stitching/dog_a.jpg', cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, dsize=(0, 0), fx=0.5, fy=0.5)
img2 = cv2.imread('/home/ej/Desktop/stitching/dog_b.jpg', cv2.IMREAD_COLOR)
img2 = cv2.resize(img2, dsize=(0, 0), fx=0.5, fy=0.5)

Optical_Flow_Farneback(img1, img2)
#Optical_Flow_DualTVL1(img1, img2)


