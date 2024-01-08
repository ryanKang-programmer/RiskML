import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

filename = './dataset.xlsx';
data = pd.read_excel(filename, engine='openpyxl')

X = data[['D mm', 't mm', 'L mm', 'd mm', 'YS MPa', 'UTS MPa', 'Exp. MPa', 'B31G MPa', 'M.B31G Mpa', 'DNV Mpa', 'Battelle Mpa', 'Shell Mpa', 'Netto Mpa',
          'School', 'Population', 'Water']].values
y = data['Result'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# SVR 모델 생성
svr_model = SVR()

# 탐색할 매개변수 그리드 정의
svr_model = SVR(kernel='linear', C=1.0)
# GridSearchCV를 사용하여 최적의 매개변수 찾기
svr_model.fit(X_train, y_train)
print("최적 변수 찾기 끝");
# 최적의 매개변수 출력

# 최적의 모델로 예측

def svm(rangePercent):
    # 평균 제곱 오차(MSE)로 성능 평가
    grid_search = svr_model;
    y_pred = grid_search.predict(X_test)
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
        if error[i] <= rangePercent:
            acceptable_error.append(error[i]);

    # 오차가 20% 이내인 데이터 개수 계산
    count = len(acceptable_error);

    # 정확도 계산
    accuracy = count / len(error)

    print("Accuracy:", accuracy)
    Results = {};
    Results['machine'] = svr_model;
    Results['accuracy'] = accuracy;
    mse = mean_squared_error(y_test, y_pred)
    Results['mse'] = mse;
    return Results;