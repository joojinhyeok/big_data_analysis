import pandas as pd
import joblib

# -------------------------------------
# 📌 1단계: 테스트 데이터 불러오기 + 전처리
# -------------------------------------

# test.csv 파일 불러오기
test = pd.read_csv('C:/csv/test.csv')

# train에서 사용한 피처만 선택
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

# 결측치가 있는 행 제거 (예측 불가능한 행 제외)
# features 리스트에 있는 컬럼들 중 하나라도 NaN(결측치)이 있으면, 해당 행(row)을 제거
test = test.dropna(subset=features)

# PassengerId는 나중에 제출 파일에서 식별자 역할 → 따로 저장
test_passenger_id = test['PassengerId']

# test[features]에 범주형 변수들(Sex, Embarked)을 숫자로 변환 (원-핫 인코딩)
# drop_first=True는 첫 번째 범주는 제거 → 다중공선성 방지
X_test = pd.get_dummies(test[features], drop_first=True)

# -------------------------------------
# 📌 2단계: 모델과 컬럼 정보 불러오기
# -------------------------------------

# 학습해둔 모델 불러오기 (.pkl로 저장된 상태)
model = joblib.load('model.pkl')

# 학습 시 사용했던 컬럼 순서 정보 불러오기 (get_dummies 순서 기준)
expected_columns = joblib.load('model_columns.pkl')

# -------------------------------------
# 📌 3단계: 컬럼 순서 맞추기 (reindex)
# -------------------------------------

# X_test의 컬럼을 expected_columns 순서에 맞추고,
# 없는 컬럼은 자동으로 0으로 채움 (fill_value=0)
# X_test의 컬럼 -> expected_columns순서에 맞춤
X_test = X_test.reindex(columns=expected_columns, fill_value=0)

# -------------------------------------
# 📌 4단계: 예측 수행 및 제출 파일 생성
# -------------------------------------

# 학습된 모델로 테스트 데이터 예측 수행
predictions = model.predict(X_test)

# PassengerId와 예측 결과(Survived)를 합쳐 제출용 DataFrame 생성
submission = pd.DataFrame({
    'PassengerId': test_passenger_id,
    'Survived': predictions
})

# 최종 결과를 submission.csv로 저장 (index=False로 인덱스 제거)
submission.to_csv('C:/csv/submission.csv', index=False)

print("✅ 제출 파일 생성 완료: C:/csv/submission.csv")
