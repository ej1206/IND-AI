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
        network = pickle.load(f)      # 신경회로망 불러오기
    return network

x, _ = get_data()                     # MNIST test data를 가져오기
network = init_network()              # 네트워크 초기화
W1, W2, W3 = network['W1'], network['W2'], network['W3']

print(x.shape)      
print(x[0].shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)