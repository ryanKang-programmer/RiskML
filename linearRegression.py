import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score;
from sklearn.linear_model import LinearRegression

tolerance = 0.5

filename = './dataset.xlsx';
data = pd.read_excel(filename, engine='openpyxl')

X = data[['Sl.', 'D mm', 't mm', 'L mm', 'd mm', 'YS MPa', 'UTS MPa', 'Exp. MPa', 'B31G MPa', 'M.B31G Mpa', 'DNV Mpa', 'Battelle Mpa', 'Shell Mpa', 'Netto Mpa']].values
y = data['Pop Mpa'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression();

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred);

match_count = 0;

volume_arr = y_test;

# 예측값과 실제값의 오차 계산
error = [];
for i in range(len(y_pred)):
    error.append(np.abs(np.abs(y_pred[i]) - np.abs(y_test[i])) / y_test[i]);

acceptable_error = [];
for i in range(len(y_pred)):
    if error[i] <= 0.1: 
        acceptable_error.append(error[i]);

match_count = len(acceptable_error);

print(score);
print("일치 개수: ", match_count)
print("일치 비율: ", match_count / len(volume_arr))