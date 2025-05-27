# 예측 및 제출 파일 생성 코드 (predict_and_submit.py 라고 저장해도 됨)

import joblib
import pandas as pd

# 모델 및 컬럼 불러오기
model = joblib.load('best_model.pkl')
expected_columns = joblib.load('model_columns.pkl')

# 테스트 데이터 불러오기
test = pd.read_csv('C:/csv/test.csv')

# 전처리 (학습 때 삭제한 컬럼 제거)
drop_cols = ['Name', 'Ticket', 'Cabin']
test = test.drop(columns=drop_cols)

# 결측치 처리
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

# 인코딩
test = pd.get_dummies(test, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# 컬럼 맞추기
X_test = test.reindex(columns=expected_columns, fill_value=0)

# 예측
predictions = model.predict(X_test)

# 제출 파일 생성
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
