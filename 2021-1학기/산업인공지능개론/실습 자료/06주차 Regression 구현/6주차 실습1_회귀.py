import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

data = np.array([[30, 12], [150,25],[300,35],[400,48],[130,21],[240,33],[350,46],[200,41],[100,20],[10,23],[190,32],[120,24],[130,19],[270,37],[244,24]])

plt.scatter(data[:, 0], data[:, 1] ) # 에 x축 y축?
plt.title("Linear Regression")
plt.xlabel("Delivery Distance")
plt.ylabel("Delivery time")
plt.axis([0,420, 0, 50])

x = data[:,0].reshape(-1, 1) # 입력 #0번재 를 가져오고, reshape 해서 1차원으로 만든다?
y = data[:,1].reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)
plt.plot(x, y_pred, color='r')
plt.show()