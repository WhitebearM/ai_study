import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

plt.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터 전처리
metro_data = pd.read_csv("dataset/metro.csv")

metro_data['timestamp'] = pd.to_datetime(metro_data['timestamp'])
metro_data['hour'] = metro_data['timestamp'].dt.hour
metro_data['day_of_week'] = metro_data['timestamp'].dt.dayofweek

metro_data = metro_data.dropna()

station_code = 150 
filtered_data = metro_data[metro_data['station_code'] == station_code]

features = ['hour', 'day_of_week', 'people_out']
target = 'people_in'
X = filtered_data[features]
y = filtered_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 특성 중요도 계산
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
plt.bar(feature_importances['Feature'], feature_importances['Importance'])
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# 3. 모델 학습
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# 4. 학습한 모델 기반 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(y_test - y_pred, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
plt.title('Prediction Error Histogram')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 5. 히트맵 생성
heatmap_data = filtered_data.groupby(['day_of_week', 'hour'])['people_in'].mean().unstack()
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".0f", cbar=True)
plt.title('Average People In by Hour and Day of Week')
plt.xlabel('Hour (0-23)')
plt.ylabel('Day of Week (0=Monday, 6=Sunday)')
plt.show()

# 6. 예측 값 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Prediction Line')
plt.title('승하차 실제값 vs 예측값')
plt.xlabel('실제 승차 인원')
plt.ylabel('예측 승차 인원')
plt.legend()
plt.grid(True)
plt.show()

# 7. 모델 예측값 기반 시간대별 시각화
hourly_mean = filtered_data.groupby('hour')['people_in'].mean()
plt.figure(figsize=(12, 6))
hourly_mean.plot(kind='bar', color='orange', alpha=0.7, label='Average People In')
plt.plot(y_test.values[:24], label='Actual People In', linestyle='-', marker='o')
plt.plot(y_pred[:24], label='Predicted People In', linestyle='--', marker='x')
plt.legend()
plt.title('시간대 별 실제값, 예측값 승차 인원')
plt.xlabel('시간')
plt.ylabel('승차 인원')
plt.show()

st.title("승차 인원 예측")

date = st.date_input("승차 일을 입력하세요.")
hour = st.number_input("시간을 입력하세요.", min_value=0 , max_value=23, value=12)
people_out = st.number_input("하차 인원을 입력하세요", min_value=0 , max_value= 10000, value=100)

day_of_week = pd.to_datetime(date).dayofweek

if st.button("예측"):
    inputData = np.array([[hour,day_of_week,people_out]])
    y_pred = rf_model.predict(inputData)
    st.write(f"예상 승객은 {int(y_pred[0])} 명 입니다.")
    