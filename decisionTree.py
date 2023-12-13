import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import math;

filename = './dataset.xlsx';
data = pd.read_excel(filename, engine='openpyxl')

X = data[['Sl.', 'D mm', 't mm', 'L mm', 'd mm', 'YS MPa', 'UTS MPa', 'Exp. MPa', 'B31G MPa', 'M.B31G Mpa', 'DNV Mpa', 'Battelle Mpa', 'Shell Mpa', 'Netto Mpa']].values
y = data['Pop Mpa'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성
model = DecisionTreeRegressor(max_depth=5)

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 예측값과 실제값의 오차 계산
error = [];
for i in range(len(y_pred)):
    error.append(np.abs(np.abs(y_pred[i]) - np.abs(y_test[i])) / y_test[i]);

acceptable_error = [];
for i in range(len(y_pred)):
    if error[i] <= 0.1: 
        acceptable_error.append(error[i]);

# 오차가 20% 이내인 데이터 개수 계산
count = len(acceptable_error);

# 정확도 계산
accuracy = count / len(error)

print("Accuracy:", accuracy)
