import math
import numpy as np
import matplotlib.pyplot as plt

# Graph 클래스
class Graph:
    def __init__(self):
        plt.figure(figsize=(8, 5))
        # y값을 저장할 리스트
        self.y = []

    def sin(self):
        x = np.arange(0, 6, 0.1)
        # 함수에 x를 입력하여 결과값 y 계산 후 리스트에 추가
        for i in x:
            y = math.sin(i)
            self.y.append(y)
        # matplotlib을 이용하여 함수 출력
        plt.xlabel('x'), plt.ylabel('y')
        plt.title("sin"), plt.plot(x, self.y)

    def cos(self):
        x = np.arange(0, 6, 0.1)
        for i in x:
            y = math.cos(i)
            self.y.append(y)
        plt.xlabel('x'), plt.ylabel('y')
        plt.title("cos"), plt.plot(x, self.y)

    def tan(self):
        x = np.linspace(-1.3, 1.3)
        for i in x:
            y = math.tan(i)
            self.y.append(y)
        plt.xlabel('x'), plt.ylabel('y')
        plt.title("tan"), plt.plot(x, self.y)

    def log(self):
        x = np.arange(0.001, 3, 0.001)
        for i in x:
            y = math.log(i)
            self.y.append(y)
        plt.xlabel('x'), plt.ylabel('y')
        plt.title("log"), plt.plot(x, self.y)


# 사용자로부터 함수 입력받기
Function = input("Enter sin cos tan log : ")

# 각 입력에 맞게 클래스의 함수 실행
graph = Graph()
if Function == "sin":
    graph.sin()
elif Function == "cos":
    graph.cos()
elif Function == "tan":
    graph.tan()
elif Function == "log":
    graph.log()
else:
    print("Enter sin cos tan log")

plt.show()





