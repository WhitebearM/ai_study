#!/usr/bin/env python
# coding: utf-8

# In[79]:


import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0 , 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

# In[80]:


#Tanh
def tanh(x):
    return np.tanh(x)

#그래프 그리기
x = np.arange(-5.0, 5.0, 0.1)
y = tanh(x)
plt.plot(x,y)
plt.ylim(-1.1, 1.1)
plt.title("Tanh Activation Function")
plt.grid()
plt.show()


# In[81]:


def relu(x):
    return np.maximum(0,x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x,y)
plt.ylim(-0.5, 5.5)
plt.title("RelU Activation function")
plt.grid()
plt.show()

# In[82]:


import tensorflow as tf
import numpy as np
import matplotlib as plt

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 데이터 준비
x_data = np.array([0,1,2,3,4], dtype=np.float32)
y_data = np.array([1,3,5,7,9], dtype=np.float32)

#모델 정의
w = tf.Variable(0.0)
b = tf.Variable(0.0)

#예측 함수
def predict(x):
    return w * x + b

#손실 함수 정의
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

#최적화 알고리즘 선택
optimizer = tf._optimizers.SGD(learning_rate = 0.01) #확률적 경사 하강법

#학습 과정
for step in range(100):
    with tf.GradientTape() as tape:
        y_pred = predict(x_data)
        loss = loss_fn(y_data, y_pred)
        
        #가중치와 편향에 대한 경사도 계산 및 업데이트
        gradients = tape.gradient(loss, [w,b])
        optimizer.apply_gradients(zip(gradients, [w,b]))
        
        if step % 10 == 0: 
            print(f"Step {step}, loss: {loss.numpy()}, w: {w.numpy()}, b:{b.numpy}")
            

# In[83]:


print("Final Parameters:", f"w={w.numpy()}, b={b.numpy()}")
print("Prediction for x = 5: ", predict(5).numpy())


# In[84]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#데이터 준비
x_data = np.array([0,1,2,3,4], dtype=np.float32)
y_data = np.array([1,3,5,7,9], dtype=np.float32)

#모델 정의
model = Sequential([
    Dense(1, input_dim = 1)
])

# In[85]:


#모델 컴파일
model.compile(optimizer='sgd', loss='mse')

# In[86]:


#모델 학습
model.fit(x_data, y_data, epochs=50, verbose=1) #50번 반복학습

# In[87]:


x_new = np.array([5], dtype=np.float32)
y_new = model.predict(x_new)

print(f"x = 5 일 때, 예측된 y = {y_new[0][0]:.2f}")

# In[100]:


import tensorflow as tf #딥러닝 라이브러리
from tensorflow.keras import layers, models #신경망 계층 및 모델 설계
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

#데이터 전처리
#MNIST 데이터셋은 이미 나눠서 제공
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[101]:


#MNIST 이미지는 28x28 크기, 채널 정보 추가 (흑백 이미지는 채널 1개)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255.0  # 0~1로 정규화
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255.0


# In[104]:


# 레이블을 One-Hot Encoding (숫자 0~9를 벡터로 변환)
# One-Hot Encoding : 범주형 데이터를 수치형 데이터로 변환하는 방법
if len(y_train.shape) == 1:  # shape이 (n,)인지 확인
    y_train = tf.keras.utils.to_categorical(y_train, 10)
if len(y_test.shape) == 1:  # shape이 (n,)인지 확인
    y_test = tf.keras.utils.to_categorical(y_test, 10)

# In[105]:


#모델 정의
model = models.Sequential() #순차적 모델 선언

# In[106]:


# 첫 번째 합성곱 계층: 필터 크기 3x3, 필터 수 32, 활성화 함수 ReLU
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
# 풀링 계층: 2x2 크기로 다운샘플링
model.add(layers.MaxPooling2D((2,2)))

# In[107]:


# 두 번째 합성곱 계층: 필터 수 64
model.add(layers.Conv2D(64,(3,3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

# In[108]:


# 세 번째 합성공 계층: 필터 수 64
model.add(layers.Conv2D(64,(3,3), activation="relu"))

# In[109]:


#Fully Connected Layer를 연결하기 위해 Flatten(압축) 처리
model.add(layers.Flatten())

# In[110]:


# 출력층: 뉴런 수 10(클래스 수), 활성화 함수 Softmax
model.add(layers.Dense(10, activation='softmax'))

# In[111]:


#모델 요약 출력(구조 확인)
model.summary()

# In[112]:


# 모델 컴파일
model.compile(optimizer='adam', #최적화 알고리즘
              loss='categorical_crossentropy', #다중 클래스 분류 손실 함수
              metrics=['accuracy']) # 평가 지표 : 정확도

# In[ ]:


# 모델 학습
history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# In[114]:


#모델 평가
test_loss, test_acc = model.evaluate(x_test,y_test)
print(f"테스트 정확도: {test_acc:.4f}")

# In[115]:


import numpy as np

sample_image = x_test[0] #첫번째 이미지 저장
sample_label = np.argmax(y_test[0]) # y_test[0] 배열에서 가장 큰 값을 가지는 요소의 인덱스를 찾아 sample_label 변수에 저장합니다.

# In[116]:


#모델로 예측 수행
predicted_label = np.argmax(model.predict(sample_image.reshape(1,28,28,1))) # 1px 너비28 높이28 마지막 1뭐지?

# In[117]:


plt.imshow(sample_image.reshape(28,28), cmap='gray')
plt.title(f"real value: {sample_label}, forcast value:{predicted_label}")
plt.show()
