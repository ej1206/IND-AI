# Pedestrian detection using HoG-SVM
import cv2
import matplotlib.pyplot as plt


image = cv2.imread("C:/Users/user/Desktop/people.jpg")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

locations, weights = hog.detectMultiScale(image)

dbg_images = image.copy()

for loc in locations:
    cv2.rectangle(dbg_images, (loc[0], loc[1]), (loc[0] + loc[2], loc[1] + loc[3]), (0, 255, 0), 2)


plt.figure(0, figsize=(12, 6))
plt.subplot(121)
plt.title('original')
plt.axis('off')
plt.imshow(image[:, :, [2, 1, 0]])

plt.subplot(122)
plt.title('detections')
plt.imshow(dbg_images[:, :, [2, 1, 0]])
plt.tight_layout()
plt.show()



