import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

filename = './dataset.xlsx';
data = pd.read_excel(filename, engine='openpyxl')

X = data[['Sl.', 'D mm', 't mm', 'L mm', 'd mm', 'YS MPa', 'UTS MPa', 'Exp. MPa', 'B31G MPa', 'M.B31G Mpa', 'DNV Mpa', 'Battelle Mpa', 'Shell Mpa', 'Netto Mpa']].values
y = data['Pop Mpa'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge 회귀 모델 생성 및 학습
alpha = 0.001  # alpha는 정규화 강도를 조절하는 매개변수, 사용자가 설정할 수 있음
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train_scaled, y_train)

# 학습된 가중치 출력
coefficients = ridge_model.coef_
intercept = ridge_model.intercept_
print("가중치:", coefficients)
print("절편:", intercept)

# 테스트 데이터에 대한 예측
y_pred = ridge_model.predict(X_test_scaled)

# 모델 평가 (평균 제곱 오차)
mse = mean_squared_error(y_test, y_pred)
print("평균 제곱 오차:", mse)

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

# 학습된 가중치 및 절편
coefficients = ridge_model.coef_
intercept = ridge_model.intercept_

# 예측 함수 정의
def ridge_regression_predict(features):
    return intercept + np.dot(features, coefficients)