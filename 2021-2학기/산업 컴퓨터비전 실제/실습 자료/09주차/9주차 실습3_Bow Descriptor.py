import cv2
import numpy as np
import matplotlib.pyplot as plt

img0 = cv2.imread('/home/ej/WorkSpace/PycharmProjects/pythonProject/image/people.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('/home/ej/WorkSpace/PycharmProjects/pythonProject/image/face.jpg', cv2.IMREAD_GRAYSCALE)

detector = cv2.ORB_create(500)
_, fea0 = detector.detectAndCompute(img0, None)
_, fea1 = detector.detectAndCompute(img1, None)

descr_type = fea0.dtype

bow_trainer = cv2.BOWKMeansTrainer(50)
bow_trainer.add(np.float32(fea0)) # add -> visual word를 여기에 둔다
bow_trainer.add(np.float32(fea1))
vocap = bow_trainer.cluster().astype(descr_type) # vocabulary를 생성

bow_descr = cv2.BOWImgDescriptorExtractor(detector, cv2.BFMatcher(cv2.NORM_HAMMING))
bow_descr.setVocabulary(vocap) # setVocabulary로 vocap을 넣어줌 bow? 를 만들어준거?

img = cv2.imread('/home/ej/WorkSpace/PycharmProjects/pythonProject/image/lena.png', cv2.IMREAD_GRAYSCALE)
kps = detector.detect(img, None)
descr = bow_descr.compute(img, kps) # Description

plt.figure(figsize=(10, 8))
plt.title('image BOW descriptor')
plt.bar(np.arange(len(descr[0])), descr[0])
plt.xlabel("vocabulary element")
plt.ylabel("frequency")
plt.tight_layout
plt.savefig("result1.png")

# bow는 유사성을 가지고 classification 해준다 / SIFT 이런건 매칭이라고 함 