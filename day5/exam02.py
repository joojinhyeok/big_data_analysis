# exam02.py
import pandas as pd
import joblib

# 1. 테스트 데이터 불러오기
test = pd.read_csv('C:/csv/test.csv')

# 2. 훈련에 사용한 피처 설정
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

# 3. 결측치 처리
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])

# 4. 원-핫 인코딩
X_test = pd.get_dummies(test[features], drop_first=True)

# 5. 학습된 모델 및 피처 컬럼 순서 불러오기
model = joblib.load('c:/python/big_data_analysis/best_model.pkl')
expected_columns = joblib.load('c:/python/big_data_analysis/model_columns.pkl')


# 6. 컬럼 순서 맞추기 (누락 컬럼은 0으로 채움)
X_test = X_test.reindex(columns=expected_columns, fill_value=0)

# 7. 예측 수행
predictions = model.predict(X_test)

# 8. 제출용 파일 생성
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

# 9. 저장
submission.to_csv('C:/csv/submission.csv', index=False)
print("submission.csv 저장 완료!")
