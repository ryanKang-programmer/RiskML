import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

tolerance = 0.30

filename = './dataset.xlsx';
data = pd.read_excel(filename, engine='openpyxl')

X = data[['Sl.', 'D mm', 't mm', 'L mm', 'd mm', 'YS MPa', 'UTS MPa', 'Exp. MPa', 'B31G MPa', 'M.B31G Mpa', 'DNV Mpa', 'Battelle Mpa', 'Shell Mpa', 'Netto Mpa']].values
y = data['Pop Mpa'].values

# 데이터 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(14,), activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# 모델 컴파일
model.compile(optimizer='Adam', loss='mean_squared_error')

# 모델 학습
model.fit(X_train, y_train, epochs=1000)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 예측값과 실제값의 오차 계산
error = [];
for i in range(len(y_pred)):
    error.append(np.abs(np.abs(y_pred[i]) - np.abs(y_test[i])) / y_test[i]);

# 오차가 20% 이내인 데이터 개수 계산
print(len(error));

acceptable_error = [];
for i in range(len(y_pred)):
    if error[i] <= 0.1: 
        acceptable_error.append(error[i]);

print(len(acceptable_error));

# 정확도 계산
accuracy = len(acceptable_error) / len(error)

print("Accuracy:", accuracy)