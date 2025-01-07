#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import streamlit2 as st
import matplotlib as plt

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

pd.options.display.float_format = '{:.2f}'.format

data = pd.read_csv("dataset/diabetes.csv")

selected_features = ['Glucose', "BMI", "Age"]
X = data[selected_features]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)


# In[5]:


joblib.dump(model,'diabetes_model.pkl')


# In[8]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')


# In[13]:


st.title('당뇨병 예측 시스템')
st.write("Glucose, BMI, Age값을 입력하여 당뇨병을 예측을 해보세요.")



# In[14]:


glucose = st.slider('Glucose(혈당수치)', min_value=0, max_value=200, value=100)
bmi = st.slider("BMI (체질량 지수)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
age = st.slider('Age (나이)', min_value=0, max_value=100, value=30)


# In[15]:


if st.button("예측하기"):
    model = joblib.load('diabetes_model.pkl')
    input_data = np.array([[glucose,bmi,age]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.write("예측 결과: 당뇨병 가능성 높")
    else:
        st.write("예측 결과: 당뇨병 가능성 낮")

