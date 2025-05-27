# exam02.py
import pandas as pd            # 판다스: 데이터를 표(데이터프레임) 형식으로 다루기 위한 라이브러리
import joblib                  # joblib: 머신러닝 모델을 파일로 저장하거나 불러오는 데 사용하는 라이브러리

# 1. 테스트 데이터 불러오기
# 예측에 사용할 데이터셋 (train.csv와 동일한 구조지만 생존 여부(Survived)는 없음)
test = pd.read_csv('C:/csv/test.csv')  # CSV 파일을 읽어와서 DataFrame 형태로 저장

# 2. 훈련에 사용한 피처(입력값으로 사용할 열) 정의
# 훈련에 사용한 컬럼과 동일하게 사용해야 모델이 올바르게 작동함
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

# 3. 결측치(NaN) 처리
# 모델은 결측값이 있는 데이터를 처리할 수 없으므로, 간단한 방식으로 채움
test['Age'] = test['Age'].fillna(test['Age'].mean())         # 'Age'가 비어있으면 평균값으로 채움
test['Fare'] = test['Fare'].fillna(test['Fare'].median())   # 'Fare'는 중앙값으로 채움
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])  # 'Embarked'는 최빈값으로 채움

# 4. 범주형 변수 인코딩 (문자를 숫자로 바꿔주는 작업)
# 'Sex', 'Embarked'와 같은 범주형 변수는 머신러닝 모델이 바로 사용할 수 없기 때문에,
# 0/1로 변환하는 원-핫 인코딩(get_dummies)을 적용
# drop_first=True는 다중공선성 방지를 위해 첫 번째 열을 제거함
X_test = pd.get_dummies(test[features], drop_first=True)

# 5. 학습된 모델과 피처 순서 불러오기
# 모델을 학습한 상태에서 저장해둔 파일(.pkl)을 다시 불러와 예측에 사용
model = joblib.load('c:/python/big_data_analysis/best_model.pkl')  # 학습 완료된 모델
expected_columns = joblib.load('c:/python/big_data_analysis/model_columns.pkl')  # 훈련 때 사용한 컬럼 순서 정보

# 6. 테스트 데이터의 컬럼 순서를 학습 데이터와 맞추기
# 모델이 학습할 때 본 컬럼과 순서가 정확히 일치해야 에러 없이 예측 가능함
# 누락된 컬럼이 있다면 0으로 채워서 문제 없이 맞춰줌
X_test = X_test.reindex(columns=expected_columns, fill_value=0)

# 7. 예측 수행
# 학습된 모델로 테스트 데이터에 대해 생존 여부 예측
# 결과는 0 또는 1의 배열 (0: 사망, 1: 생존)
predictions = model.predict(X_test)

# 8. 제출용 데이터프레임 생성
# 예측 결과를 PassengerId와 함께 묶어서 CSV로 만들 준비
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],  # 식별자 컬럼
    'Survived': predictions              # 예측 결과
})

# 9. 결과를 CSV 파일로 저장
# index=False는 불필요한 인덱스 숫자 열을 제거하고 저장
submission.to_csv('C:/csv/submission.csv', index=False)

# 완료 메시지 출력
print("submission.csv 저장 완료!")  # 콘솔에 출력
