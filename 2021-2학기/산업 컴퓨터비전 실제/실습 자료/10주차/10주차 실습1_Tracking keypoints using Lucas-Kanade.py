#실행 해보면 outlier가 별로 없음
# color, 밝기가 별로 다르지 않다 / 움직임이 크지 않다. -> 2가지 가정을 만족하기 때문에

import cv2
import numpy as np

video = cv2.VideoCapture('C:/Users/user/Desktop/traffic.mp4')
prev_pts = None
prev_gray_frame = None
tracks = None

while(True):
    retval, frame = video.read()
    if not retval: break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_pts is not None:
        pts, status, errors = cv2.calcOpticalFlowPyrLK( # lucas kanade를 이용
            prev_gray_frame, gray_frame, prev_pts, None, winSize=(15, 15), maxLevel=5, # 피라미드를 얼마나 쌓을지
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            # 언제끝날지 
        good_pts = pts[status == 1] # status가 1인게 제대로 된거, 0이면 안좋은거
        if tracks is None:
            tracks = good_pts

        else: tracks = np.vstack((tracks, good_pts))
        for p in tracks:
            cv2.circle(frame, (p[0], p[1]), 3, (0, 255, 0), -1)

    else:
        pts = cv2.goodFeaturesToTrack(gray_frame, 500, 0.05, 10) # 특징찾기
        pts = pts.reshape(-1, 1, 2)

    prev_pts = pts # 현재 pts에 넣어준다.
    prev_gray_frame = gray_frame

    cv2.imshow('frame', frame)
    key = cv2.waitKey() & 0xff

    if key == 27:
        break
    if key == ord('c'):  ## esc, c가 아닌 다른 키를 눌러야 한다.
        tracks = None
        prev_pts = None

cv2.destroyAllWindows()
