import numpy as np

def indentity_function(x):
  return x

def sigmoid(x):
  return 1/(1 + np.exp(-x))


def init_network():
  # 가중치, 편향 ***초기화하고 얘네를 딕셔너리 변수인 network에 저장
  # 딕셔너리 network에는 각 층에 필요한 매개변수(가중치, 편향) 저장
  network = {}
  network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
  network['b1'] = np.array([0.1,0.2,0.3])
  network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
  network['b2'] = np.array([0.1,0.2])
  network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
  network['b3'] = np.array([0.1,0.2])
  return network

def forward(network, x):
  # forward는 입력신호를 출력신호로 변환하는 처리 과정 구현
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x,W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = indentity_function(a3) # y=a3
  return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)