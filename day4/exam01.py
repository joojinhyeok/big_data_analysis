import pandas as pd

# CSV 파일 불러오기
train = pd.read_csv('C:/csv/train.csv')
test = pd.read_csv('C:/csv/test.csv')

print("원본 train 데이터 shape:", train.shape)
print("원본 test 데이터 shape:", test.shape)

"""
📌 전처리 요약
1. 결측치 채우기 (Age, Embarked, Fare)
2. 이상치 제거 (Fare 기준 IQR 방식)
3. 범주형 변수 인코딩 (Sex, Embarked → 숫자형으로 변환)
"""

# -------------------------------
# ✅ 1. 결측치 채우기
# -------------------------------

# Age 컬럼 결측치를 평균값으로 채우기
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)

# Embarked 컬럼 결측치를 최빈값으로 채우기 (보통 'S')
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

# test 데이터의 Fare 컬럼 결측치는 중앙값으로 채우기
test['Fare'].fillna(test['Fare'].median(), inplace=True)

print("\n✅ 결측치 채우기 완료")
print("train 결측치 수:\n", train.isnull().sum())
print("test 결측치 수:\n", test.isnull().sum())

# -------------------------------
# ✅ 2. 이상치 제거 (Fare 기준 IQR 방식)
# -------------------------------

# Q1: 25%, Q3: 75% 분위수 구하기
Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1

# 이상치 판단 기준값 계산
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Fare 값이 정상 범위 안에 있는 데이터만 남기기
train = train[(train['Fare'] >= lower_bound) & (train['Fare'] <= upper_bound)]

print("\n✅ 이상치 제거 완료 (Fare 기준)")
print("이상치 제거 후 train 데이터 shape:", train.shape)

# -------------------------------
# ✅ 3. 범주형 변수 인코딩 (get_dummies)
# -------------------------------

# 범주형 컬럼: Sex, Embarked → 숫자형으로 변환
train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)

# train에는 있고 test에는 없는 컬럼은 0으로 채워 넣기 (동일한 컬럼 구조 맞추기 위함)
for col in train.columns:
    if col not in test.columns and col != 'Survived':
        test[col] = 0

# 컬럼 순서 맞추기 (예측 때 오류 방지)
test = test[train.drop('Survived', axis=1).columns]

print("\n✅ 인코딩 및 컬럼 정리 완료")
print("train 컬럼 목록:", train.columns.tolist())
print("test 컬럼 목록:", test.columns.tolist())


# -------------------------------
# ✅ 4. 예측에 사용할 컬럼 선택 (Feature Selection)
# -------------------------------

# 모델에 입력할 컬럼(특징)들을 리스트로 정의
# 선택 이유:
# - Pclass: 객실 등급 (1등실/2등실/3등실)
# - Age: 나이
# - Fare: 티켓 가격
# - Sex_male: 남성이면 1, 여성이면 0 (get_dummies로 변환됨)
features = ['Pclass', 'Age', 'Fare', 'Sex_male']

# train 데이터에서 입력 변수(X)만 따로 추출
X = train[features]

# train 데이터에서 정답값(y, 즉 생존 여부)만 따로 추출
y = train['Survived']

# 확인용 출력 (현재 모델에 들어갈 컬럼과 데이터 크기 확인)
print("🎯 선택된 특징 컬럼:\n", features)
print("X shape:", X.shape)   # (행 개수, 입력 변수 개수)
print("y shape:", y.shape)   # (행 개수,)


# -------------------------------
# ✅ 5. 로지스틱 회귀 모델 학습
# -------------------------------

# 🎯 필요한 라이브러리 불러오기
# - LogisticRegression: 분류 문제를 위한 모델 (생존/사망 같은 이진 분류에 적합)
# - train_test_split: 데이터를 학습용/검증용으로 나누기
# - accuracy_score: 예측이 얼마나 맞았는지 평가하는 함수
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🎯 학습용/검증용 데이터 분할
# - X: 입력 데이터 (특징들)
# - y: 정답 (Survived)
# - test_size=0.2 → 전체의 20%를 검증용으로 사용
# - random_state=42 → 항상 같은 결과를 위해 랜덤 시드 고정
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🎯 로지스틱 회귀 모델 생성
# - max_iter=1000 → 반복 횟수를 충분히 늘려서 학습이 잘 되도록 함
model = LogisticRegression(max_iter=1000)

# 🎯 학습 데이터(X_train, y_train)를 이용해서 모델 훈련
model.fit(X_train, y_train)

# 🎯 검증 데이터(X_val)를 이용해서 예측 수행
y_pred = model.predict(X_val)

# 🎯 예측값(y_pred)과 실제값(y_val)을 비교해서 정확도 계산
accuracy = accuracy_score(y_val, y_pred)

# 🎯 결과 출력
print("\n✅ 모델 학습 및 예측 완료")
print("Validation Accuracy (검증 정확도): {:.2f}%".format(accuracy * 100))


# -------------------------------
# ✅ 6. 실제 test 데이터로 예측 + 결과 저장
# -------------------------------

# 🎯 test 데이터는 이미 전처리와 인코딩이 완료된 상태
# → 우리가 선택한 특징(features)만 추출
X_test = test[features]  # features = ['Pclass', 'Age', 'Fare', 'Sex_male']

# 🎯 훈련된 모델을 사용해 test 데이터에 대해 예측 수행
# → 예측 결과는 0 또는 1로 이루어진 배열
test_predictions = model.predict(X_test)

# 🎯 test 데이터에 'Survived' 컬럼을 새로 만들어서 예측 결과 넣기
test['Survived'] = test_predictions

# 🎯 최종 제출 파일은 'PassengerId'와 'Survived'만 포함
submission = test[['PassengerId', 'Survived']]

# 🎯 CSV 파일로 저장 (index=False: 인덱스 컬럼 제거)
submission.to_csv('submission.csv', index=False)

# 🎯 확인용 출력
print("\n📄 예측 결과 파일 저장 완료! (submission.csv)")
print(submission.head())
