import pandas as pd  # 판다스: 데이터프레임(표 형태 데이터) 처리 라이브러리
from sklearn.model_selection import train_test_split, GridSearchCV  # 데이터 분리 & 하이퍼파라미터 튜닝
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델 (이진 분류용)
from sklearn.metrics import accuracy_score  # 모델 성능 평가용 정확도 측정 함수
import joblib  # 모델 저장 및 불러오기용 라이브러리

# 1. 학습용 데이터 불러오기 (train.csv는 생존 여부가 포함된 학습 데이터셋)
train = pd.read_csv('C:/csv/train.csv')  # CSV 파일 읽기

# 2. 사용할 피처(입력 변수) 정의 및 결측치 제거
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
# 위 피처들 중 하나라도 NaN이 있으면 해당 행은 제거
train = train.dropna(subset=features)

# 3. 범주형 변수 인코딩
# 'Sex'와 'Embarked'처럼 문자열로 되어 있는 컬럼을 숫자로 변환해야 모델이 이해할 수 있음
# get_dummies: 원-핫 인코딩 수행
# drop_first=True는 첫 번째 범주를 제거해 다중공선성 문제 방지
X = pd.get_dummies(train[features], drop_first=True)

# 타겟값(정답): 생존 여부(Survived 컬럼)
y = train['Survived']

# 4. 학습용 데이터와 검증용 데이터 분리
# train_test_split: 데이터를 8:2 비율로 나눠 학습/검증용으로 사용
# random_state: 랜덤 시드 고정 → 결과 재현 가능
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 로지스틱 회귀 모델 정의 및 하이퍼파라미터 튜닝
# C: 규제 강도(작을수록 강한 규제), max_iter: 반복 횟수
params = {'C': [0.01, 0.1, 1, 10], 'max_iter': [100, 500, 1000]}
# GridSearchCV: 파라미터 조합을 모두 시험해보고 가장 좋은 결과를 찾아줌 (교차검증 cv=5 사용)
grid = GridSearchCV(LogisticRegression(), param_grid=params, cv=5)
grid.fit(X_train, y_train)  # 모델 학습

# 6. 최적의 결과 출력
print("Best Params:", grid.best_params_)  # 가장 성능 좋은 파라미터 조합 출력
model = grid.best_estimator_              # 최적의 모델 추출
y_pred = model.predict(X_val)             # 검증용 데이터에 대해 예측 수행
print("Accuracy:", accuracy_score(y_val, y_pred))  # 정확도 계산 및 출력

# 7. 학습된 모델과 피처 목록 저장
# 이후 예측 스크립트(exam02.py 등)에서 이 모델을 다시 불러와 사용할 수 있음
joblib.dump(model, 'best_model.pkl')  # 모델 저장
joblib.dump(X.columns.tolist(), 'model_columns.pkl')  # 학습에 사용한 피처 순서도 함께 저장
print("모델과 피처 정보 저장 완료!")

# (선택) 현재 작업 디렉토리 및 파일 목록 출력 – 경로 오류 점검용
import os
print("현재 경로:", os.getcwd())
print("현재 폴더 안의 파일들:", os.listdir())
