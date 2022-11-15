'''
Gabor Filter : 가운데가 밝고, 가장자리가 어두움 -> 가운데 부분의 edge가 잘 드러남
                  얼굴인식에 많이 사용됐었음


Discrete Fourier Transform -> 매그니튜드 계산
 shift 하고 magnitude 구함 -> 근데 매그니튜드 편차가 너무 심해서 로그 취해준다
                                          log 안취하면 거의 안보임. 가운데값이 너무 커서 검정색 배경 가운데에 하얀 점 하나만 보임

'''

# 3) Frequency-based Filtering ** 오늘배운것중에 제일 많이담고있는것 같음 DFT 기반의 필터링
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("C:/Users/user/PycharmProjects/test1/lena.png", 0).astype(np.float32) / 255


fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0, 1])
sz = 25

mask = np.zeros(fft.shape, np.uint8)
mask[image.shape[0]//2-sz:image.shape[0]//2+sz, image.shape[1]//2-sz:image.shape[1]//2+sz, :] = 1
fft_shift *= mask
fft = np.fft.ifftshift(fft_shift, axes=[0, 1])

filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
mask_new = np.dstack((mask, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)))

plt.figure()
plt.subplot(131)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap='gray')

plt.subplot(132)
plt.axis('off')
plt.title('no high frequencies')
plt.imshow(filtered, cmap='gray')

plt.subplot(133)
plt.axis('off')
plt.title('mask')
plt.imshow(mask_new*255, cmap='gray')
plt.tight_layout
plt.show()