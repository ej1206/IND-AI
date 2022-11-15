
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)


x = np.arange(-5.0, 5.0, 0.1) # arrnage :  -0.5~0.5까지 0.1 간격으로 numpy 배열 생성
y = step_function(x)
print(y)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)           # y축의 범위 지정
plt.show()