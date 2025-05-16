# 필요한 라이브러리 불러오기
import pandas as pd  # 데이터프레임을 다루기 위한 라이브러리
from sklearn.model_selection import train_test_split, GridSearchCV  # 데이터 분할 및 파라미터 튜닝 도구
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델
from sklearn.metrics import accuracy_score  # 모델 평가용 정확도 계산 함수

# 1. 학습용 데이터 불러오기 (train.csv는 타이타닉 탑승자 정보가 담긴 파일)
train = pd.read_csv('C:/csv/train.csv')  # CSV 파일을 데이터프레임으로 읽음

# 2. 사용할 열(컬럼)을 고르고, 결측치(NaN)가 있는 행은 제거
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']  # 사용할 피처(입력 변수)들
train = train.dropna(subset=features)  #  Pclass, Sexm Age, Fare, Embarked중 하나라도 NaN이면 해당 행 제거 

# 범주형 변수(Sex, Embarked)를 숫자로 바꾸기 위해 One-Hot Encoding 수행
# drop_first=True는 중복 방지를 위해 하나의 더미 열을 제거
X = pd.get_dummies(train[features], drop_first=True)

# 정답값(타겟값) y는 'Survived' 컬럼 (0 = 생존 X, 1 = 생존 O)
y = train['Survived']

# 3. 학습용 데이터와 검증용 데이터로 나누기
# X: 입력 데이터, y: 정답 데이터
# test_size=0.2 → 80%는 학습, 20%는 검증용으로 사용
# random_state=42 → 결과 재현을 위해 랜덤 시드 고정
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 하이퍼파라미터 후보 설정 및 최적값 찾기 위한 그리드 서치 수행
# LogisticRegression에 대해 C(규제 정도), max_iter(반복 횟수)를 튜닝
params = {'C': [0.01, 0.1, 1, 10], 'max_iter': [100, 500, 1000]}

# GridSearchCV: 모든 조합을 시도해 보고 성능이 가장 좋은 파라미터를 선택해줌
# cv=5 → 데이터를 5등분해서 교차검증 수행 (과적합 방지)
grid = GridSearchCV(LogisticRegression(), param_grid=params, cv=5)

# 학습 데이터로 그리드 서치 실행 → 여러 파라미터 조합으로 모델을 훈련
grid.fit(X_train, y_train)

# 5. 최적의 하이퍼파라미터 출력
print("** Best Parameters **:", grid.best_params_)

# 최적의 파라미터로 학습된 모델 꺼내오기
model = grid.best_estimator_

# 6. 검증용 데이터에 대해 예측 수행
y_pred = model.predict(X_val)

# 실제 정답(y_val)과 비교해서 정확도 평가
print("Accuracy:", accuracy_score(y_val, y_pred))  # 0~1 사이 숫자로 정확도 표시
