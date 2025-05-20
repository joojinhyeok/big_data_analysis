import pandas as pd
# -------------------------------------
# 1단계: 테스트 데이터 불러오기 + 전처리
# -------------------------------------
# test.csv 파일 불러오기
test = pd.read_csv('C:/csv/test.csv')

# train에서 사용한 피처만 사용
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

# 결측치 제거
test = test.dropna(subset=features)

# PassengerId는 제출 파일 만들 때 사용하니까 따로 저장(db에서의 pk역할)
test_passenger_id = test['PassengerId']

# 범주형 피처를 수치형으로 변환 (get_dummies)
X_test = pd.get_dummies(test[features], drop_first=True)

# print(X_test.head()) -> 확인용

# -------------------------------
# 2단계: 모델과 컬럼 정보 불러오기
# -------------------------------
import joblib

# 1. 저장된 모델 불러오기
model = joblib.load('model.pkl')    # 경로는 필요에 따라 수정

# 2. 학습 당시 사용된 컬럼 리스트 불러오기
expected_columns = joblib.load('model_columns.pkl')

# -------------------------------
# 3단계: 컬럼 순서 맞추기(reindex)
# -------------------------------
# X_test의 컬럼 순서 맞추기
X_test = X_test.reindex(columns=expected_columns, fill_value=0)

# -----------------------------------
# 4단계: 예측 수행 및 제출 파일 생성
# -----------------------------------
# 1. 예측 수행
predictions = model.predict(X_test)

# 2. 예측 결과를 PassengerId와 함께 DataFrame으로 만들기
submission = pd.DataFrame({
    'PassengerId': test_passenger_id,
    'Survived': predictions
})

# 3. 제출용 csv 파일로 저장
submission.to_csv('C:/csv/submission.csv', index=False)
