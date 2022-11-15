import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/home/ej/WorkSpace/PycharmProjects/pythonProject/image/lena.png').astype(np.float32) / 255
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab) # 사람의 시각과 비슷한 lab으로 변환

data = image_lab.reshape((-1, 3))

num_classes = 8  ### 바꿔가면서 실험해보기 / 15로 하면 원본과 거의 비슷 / 2로 하면 부자연스럽당 거의 명암 수준
critera = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1) # 어떤 기준으로 해줄지
_, labels, centers = cv2.kmeans(data, num_classes, None, critera, 10, cv2.KMEANS_RANDOM_CENTERS) # random center로 지정했기 때문에 num class를 2로 하면 계속 다른 결과가 나올 수 있다 
segmented_lab = centers[labels.flatten()].reshape(image.shape)
segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_Lab2RGB)

plt.subplot(121)
plt.axis('off'), plt.title('original')
plt.imshow(image[:, :, [2, 1, 0]])
plt.subplot(122)
plt.axis('off'), plt.title('segmented')
plt.imshow(segmented)
plt.savefig("test1.png")