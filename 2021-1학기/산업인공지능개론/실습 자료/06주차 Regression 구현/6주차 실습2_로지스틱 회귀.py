import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

dataset = pd.read_csv('User_Data.csv')

x = dataset.iloc[:, [2,3]].values # 2 3번 애들을 입력
y = dataset.iloc[:, 4].values    # 4번 애들을 출력으로?

from sklearn.model_selection import train_test_split # train test set 나누겠다.
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.25, random_state=0) # testsize는 전체의 25%

from sklearn.preprocessing import StandardScaler # AGE는 값들 비슷한데 salary는 편차가 너무 심하니까 일정 범위 내 값으로 바꿔준다.
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
print(xtrain[0:10, :]) # 바꾼거 10개만 출력

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)
print("혼동 행렬 : \n", cm)

from sklearn.metrics import accuracy_score
print("정확도 : ", accuracy_score(ytest, y_pred))