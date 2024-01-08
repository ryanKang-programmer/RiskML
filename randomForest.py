import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

filename = './dataset.xlsx';
data = pd.read_excel(filename, engine='openpyxl');

X = data[['D mm', 't mm', 'L mm', 'd mm', 'YS MPa', 'UTS MPa', 'Exp. MPa', 'B31G MPa', 'M.B31G Mpa', 'DNV Mpa', 'Battelle Mpa', 'Shell Mpa', 'Netto Mpa',
          'School', 'Population', 'Water']].values
y = data['Result'].values

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);

model = RandomForestRegressor(n_estimators=10, max_depth=5)

# 모델 학습
model.fit(X_train, y_train)

def randomForest(rangePercent):
    # 테스트 데이터에 대한 예측
    y_pred = model.predict(X_test);
    # 예측값과 실제값의 오차 계산
    error = [];
    for i in range(len(y_pred)):
        error.append(np.abs(np.abs(y_pred[i]) - np.abs(y_test[i])) / y_test[i]);

    acceptable_error = [];
    for i in range(len(y_pred)):
        if error[i] <= rangePercent:
            acceptable_error.append(error[i]);

    # 정확도 계산
    accuracy = len(acceptable_error) / len(y_pred)

    print("Accuracy:", accuracy)
    importances = model.feature_importances_
    print(importances)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Coefficient of Determination (R^2):", r2)
    print("Median Absolute Error (MedAE):", medae)

    # # 정확도(Accuracy) 계산
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy: {:.2f}".format(accuracy))

    # # 정밀도(Precision) 계산
    # precision = precision_score(y_test, y_pred)
    # print("Precision: {:.2f}".format(precision))

    # # 재현율(Recall) 계산
    # recall = recall_score(y_test, y_pred)
    # print("Recall: {:.2f}".format(recall))

    # # F1-score 계산
    # f1 = f1_score(y_test, y_pred)
    # print("F1-score: {:.2f}".format(f1))

    # # ROC AUC 계산
    # roc_auc = roc_auc_score(y_test, y_pred)
    # print("ROC AUC: {:.2f}".format(roc_auc))
    Results = {};
    Results['machine'] = model;
    Results['accuracy'] = accuracy;
    mse = mean_squared_error(y_test, y_pred)
    Results['mse'] = mse;
    return Results;