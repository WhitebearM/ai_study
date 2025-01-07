#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

plt.rcParams['font.family'] = "Malgun Gothic"

data = pd.read_csv("dataset/arugula2.csv")


data = data.dropna()

X = data[['\naverage temperature', '\nhighest temperature','\nHighest-lowest temperature','\nchoejeogion\nlowest temperature']]
y = data['average price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)



# In[35]:


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X,y)

plt.figure(figsize=(10,6))
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(x= "importance", y='feature', data=importances)
plt.title("특성중요도")
plt.show()


# In[30]:


plt.figure(figsize=(12,18))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("히트맵")
plt.show()


# In[33]:


plt.figure(figsize=(10,6))
sns.histplot(y, kde=True)
plt.title('평균가격 히스토그램')
plt.xlabel("평균가격")
plt.show()

