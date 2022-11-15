import cv2
import argparse

# Parser 라이브러리
parser = argparse.ArgumentParser()
parser.add_argument('--path', default="C:/Users/user/PycharmProjects/test1/Eunjung.jpg")
params = parser.parse_args()

# imread - gray 이미지 로드
gray = cv2.imread(params.path, 0)

# imshow - gray 이미지 출력
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

