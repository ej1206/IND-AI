import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow import keras

# 데이터 로드
DATA = pd.read_csv('0507.csv', encoding='utf-8')

# 데이터 확인
#print(DATA.head())

# Train 및 Test Dataset 생성
data = DATA.drop('Class', axis=1).values
target = DATA['Class'].values
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, stratify=target, random_state=0)

# 모델 함수 생성
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(len(y_train), activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

# Trainset을 이용하여 학습
model = create_model()
model.fit(x_train, y_train, epochs=20)

# Testset을 이용하여 accuracy 계산
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\ntest accuracy:', test_acc)

########################## 학습 결과 (각 epoch 당 loss 및 accuracy) ##########################

'''
Epoch 1/20
2527/2527 [==============================] - 59s 23ms/step - loss: 2.4974 - accuracy: 0.6944
Epoch 2/20
2527/2527 [==============================] - 59s 24ms/step - loss: 0.1748 - accuracy: 0.9390
Epoch 3/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1309 - accuracy: 0.9526
Epoch 4/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1209 - accuracy: 0.9589
Epoch 5/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1179 - accuracy: 0.9611
Epoch 6/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1152 - accuracy: 0.9611
Epoch 7/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1163 - accuracy: 0.9618
Epoch 8/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1179 - accuracy: 0.9604
Epoch 9/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1132 - accuracy: 0.9621
Epoch 10/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1223 - accuracy: 0.9601
Epoch 11/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1174 - accuracy: 0.9615
Epoch 12/20
2527/2527 [==============================] - 57s 23ms/step - loss: 0.1198 - accuracy: 0.9604
Epoch 13/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1173 - accuracy: 0.9614
Epoch 14/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1208 - accuracy: 0.9607
Epoch 15/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1189 - accuracy: 0.9603
Epoch 16/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1172 - accuracy: 0.9623
Epoch 17/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1186 - accuracy: 0.9620
Epoch 18/20
2527/2527 [==============================] - 58s 23ms/step - loss: 0.1195 - accuracy: 0.9606
Epoch 19/20
2527/2527 [==============================] - 61s 24ms/step - loss: 0.1194 - accuracy: 0.9607
Epoch 20/20
2527/2527 [==============================] - 59s 23ms/step - loss: 0.1175 - accuracy: 0.9613
1083/1083 - 11s - loss: 0.1182 - accuracy: 0.9577

test accuracy: 0.9577403664588928
'''
