import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

filename = './dataset.xlsx';
data = pd.read_excel(filename, engine='openpyxl')

X = data[['Sl.', 'D mm', 't mm', 'L mm', 'd mm', 'YS MPa', 'UTS MPa', 'Exp. MPa', 'B31G MPa', 'M.B31G Mpa', 'DNV Mpa', 'Battelle Mpa', 'Shell Mpa', 'Netto Mpa']].values
y = data['Pop Mpa'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVR 모델 생성
svr_model = SVR()

# 탐색할 매개변수 그리드 정의
param_grid = {'C': [10000], 'kernel': ['linear'], 'epsilon': [1, 2]}

# GridSearchCV 객체 생성
grid_search = GridSearchCV(svr_model, param_grid, cv=5)

# GridSearchCV를 사용하여 최적의 매개변수 찾기
grid_search.fit(X_train, y_train)

# 최적의 매개변수 출력
print(f"Best Parameters: {grid_search.best_params_}")

# 최적의 모델로 예측
y_pred = grid_search.predict(X_test)

# 평균 제곱 오차(MSE)로 성능 평가
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# R 제곱 계산
r_squared = r2_score(y_test, y_pred)
print(f"R-squared: {r_squared:.4f}")

# 평균 절대 오차 계산
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

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
