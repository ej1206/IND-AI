import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from my_function import softmax, sigmoid
import pickle
from PIL import Image

def get_data():
    (x_train, t_train), (x_test, t_test) =  load_mnist(flatten=True, normalize=False,one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)       # 학습된 신경회로망 불러오기
    return network

# 출력측 / 은닉층 2층 / 출력층 신경회로 망 예측하는 부분.
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()                     # MNIST test data를 가져온다
network = init_network()              # *** 네트워크 초기화

arruracy_cnt = 0                      # 0으로 초기화
for i in range(len(x)):               # 지금 테스트중. test 데이터는 10000장
    y = predict(network, x[i])
    p = np.argmax(y)                  # 확률이 가장 높은 원소의 인덱스 계산
    if p == t[i]:                     # 실제 답과 비교
        arruracy_cnt += 1             # 정답 개수 추가

print("Accuracy:" + str(float(arruracy_cnt) / len(x))) # Accuracy 계산 >> 0.9352
