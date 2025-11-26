# [문제 2] 2유형 - 분류 모델링 (정확도 채점)
# 데이터셋: train.csv, test.csv 목표: 승객의 **생존 여부(0 또는 1)**를 예측하여 제출하시오. 평가지표: Accuracy (정확도)

# 제출 조건:
# 파일명: result.csv
# 컬럼: PassengerId, Survived (반드시 0 또는 1의 정수 값이어야 함)

import pandas as pd

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

# 1. 데이터 유형 파악
# print(train.info()) -> Age, Cabin, Embarked에 결측치 존재 
# print(test.info()) -> Age, Fare, Cabin에 결측치 존재
# print(train.head())

# 2. 데이터 전처리
p_id = test['PassengerId']

# 2-1 데이터 셋 분리
X_train = train.drop(['PassengerId', 'Survived', 'Name', 'Cabin', 'Embarked', 'Ticket'], axis=1)
y = train['Survived']
X_test = test.drop(['PassengerId', 'Name', 'Cabin', 'Embarked', 'Ticket'], axis=1)

# 2-2 결측치 처리
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
X_test['Age'] = X_test['Age'].fillna(X_train['Age'].mean())

X_test['Fare'] = X_test['Fare'].fillna(X_train['Fare'].mean())

# 2-3 수치형 변수 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

num_columns = X_train.select_dtypes(exclude='object').columns

X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
X_test[num_columns] = scaler.transform(X_test[num_columns])

# 2-4 범주형 변수 인코딩
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

X_train['Sex'] = encoder.fit_transform(X_train['Sex'])
X_test['Sex'] = encoder.transform(X_test['Sex'])

# 3. 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2)

# 4. 모델링
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

# 5. 평가지표 검증
# 평가지표: Accuracy (정확도)
from sklearn.metrics import accuracy_score 
# import sklearn.metrics
# print(dir(sklearn.metrics))
score = accuracy_score(y_val, y_pred)
print(score)

# 6. 저장 및 제출
# 제출 조건:
# 파일명: result.csv
# 컬럼: PassengerId, Survived (반드시 0 또는 1의 정수 값이어야 함)
sur = model.predict(X_test)
result = pd.DataFrame({
    'PassengerId': p_id,
    'Survived': sur
})

result.to_csv('result.csv', index=False)

# print(result)