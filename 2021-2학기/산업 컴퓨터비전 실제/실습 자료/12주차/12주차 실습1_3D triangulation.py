# 3D triangulation # 랜덤하게 점 만들어서 실습
import cv2
import numpy as np

p1 = np.eye(3, 4, dtype=np.float32)
p2 = np.eye(3, 4, dtype=np.float32)

p2[0, 3] = -1 # p2를 움직여준다.

N = 5

#3D 포인트 랜덤하게 5개 생성
points3d = np.empty((4, N), np.float32)
points3d[:3, :] = np.random.randn(3, N)
points3d[3, :] = 1

# p1 x points3d 해서 points1 만든다
points1 = p1 @ points3d
points1 = points1[:2, :] /points1[2, :]
points1[:2, :] += np.random.randn(2, N) * 1e-2

# points3d를 프로젝션해서 points2 생성
points2 = p2 @ points3d
points2 = points2[:2, :] /points2[2, :]
points2[:2, :] += np.random.randn(2, N) * 1e-2 # 이런거 곱하는거 : 노이즈 생성
                                               # 노이즈 커질수록 값이 달라진당당구리

points3d_reconstr = cv2.triangulatePoints(p1, p2, points1, points2)
points3d_reconstr /= points3d_reconstr[3, :]

print('Original points')
print(points3d[:3].T)
print('Reconstructed points')
print(points3d_reconstr[:3].T)