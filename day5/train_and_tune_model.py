import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib  # 모델 저장용

# 1. 학습용 데이터 불러오기
train = pd.read_csv('C:/csv/train.csv')

# 2. 사용할 피처 설정 및 결측치 제거
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
train = train.dropna(subset=features)

# 3. 범주형 피처 인코딩
X = pd.get_dummies(train[features], drop_first=True)
y = train['Survived']

# 4. 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 모델 정의 및 하이퍼파라미터 튜닝
params = {'C': [0.01, 0.1, 1, 10], 'max_iter': [100, 500, 1000]}
grid = GridSearchCV(LogisticRegression(), param_grid=params, cv=5)
grid.fit(X_train, y_train)

# 6. 결과 출력
print("Best Params:", grid.best_params_)
model = grid.best_estimator_
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))

# ✅ 7. 학습된 모델과 피처 목록 저장
joblib.dump(model, 'best_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print("모델과 피처 정보 저장 완료!")

import os
print("현재 경로:", os.getcwd())
print("현재 폴더 안의 파일들:", os.listdir())