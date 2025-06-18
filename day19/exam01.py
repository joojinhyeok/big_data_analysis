# -------------------------------------------------------------------------------------
# 2유형 흐름
# 1. 데이터 확인
# 2. 데이터 전처리 -> 결측치를 '최빈값'이나 '평균값', '중앙값'을 넣어보면서 모델 평가 해보기
# 3. 데이터 분할
# 4. 모델링 및 학습
# 5. 성능평가
# 6. 예측 결과 제출 및 확인
# -------------------------------------------------------------------------------------
# 분류 문제(ex: 생존, 구매여부 등) - RandomForestClassifier
# 회귀 문제(ex: 총구매액, 보험료 등) - RandomForestRegressor
# -------------------------------------------------------------------------------------
# [분류 문제 유형] - Titanic 생존 여부 예측
#
# Titanic 탑승객 데이터가 주어졌을 때, 탑승객의 생존 여부(Survived)를 예측하는
# 분류 모델을 생성하시오. test.csv에 대한 예측 결과를 result.csv로 저장하시오.
#
# [데이터 설명]
# - train.csv: 훈련 데이터 (생존 여부 포함)
# - test.csv: 테스트 데이터 (생존 여부 없음)
# - 예측 대상 컬럼: Survived (0 = 사망, 1 = 생존)
#
# [요구사항]
# 1. train.csv와 test.csv를 불러오시오.
# 2. 다음 컬럼은 예측에 사용하지 않으므로 제거하시오:
#    - 'PassengerId', 'Name', 'Ticket', 'Cabin'
# 3. 결측치는 다음 기준으로 처리하시오:
#    - 'Age' → 평균값으로 채움
#    - 'Embarked' → 최빈값으로 채움
#    - test의 'Fare' → 평균값으로 채움
# 4. 범주형 변수 'Sex', 'Embarked'는 LabelEncoder를 이용하여 수치형으로 변환하시오.
# 5. train 데이터를 X, y로 분리하시오 (y는 'Survived' 컬럼)
# 6. X, y를 학습용과 평가용으로 분할하시오 (test_size=0.3, random_state=42)
# 7. 모델은 RandomForestClassifier를 사용하시오.
#    - n_estimators=100, max_depth=6, random_state=42
# 8. 평가 지표는 accuracy_score를 사용하여 출력하시오.
# 9. test.csv에 대한 예측 결과를 result.csv로 저장하시오.
#    - 컬럼명은 'pred'로 하고, 인덱스는 포함하지 마시오.
# -------------------------------------------------------------------------------------

import pandas as pd

# 데이터 읽어오기
train = pd.read_csv('C:/csv/train.csv')
test = pd.read_csv('C:/csv/test.csv')

# 데이터 확인
# print(train.info()) -> 수치형: Age컬럼 결측치o, 범주형: Embarked 결측치o, Name/Sex도/Ticket 범주형
# print(test.info()) -> 수치형: Age, Fare컬럼 결측치o, 범주형:  Name/Sex/Ticket/Embarked 범주형

# 데이터 전처리

# 예측에 필요없는 데이터 제거
train = train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
test = test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# 결측치 처리
train['Age'] = train['Age'].fillna(train['Age'].mean())  # 평균값으로 Age 결측치 채우기   
test['Age'] = test['Age'].fillna(test['Age'].mean())  # 평균값으로 Age 결측치 채우기   

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

# 인코딩
from sklearn.preprocessing import LabelEncoder
# print(help(sklearn))

le = LabelEncoder()

train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train['Embarked'] = le.fit_transform(train['Embarked'])
test['Embarked'] = le.transform(test['Embarked'])

# 데이터 분할
X = train.drop(columns='Survived')
y = train['Survived']

# 학습용 평가용으로 분할(test_size=0.3, random_state=42)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델링 및 학습
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)
pred1 = model.predict(X_val)
# print(pred1)

# 성능평가
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_val, pred1)    # (실제값, 예측값)
# print(acc)

# 테스트 데이터 예측 및 저장
test_X = test
pred2 = model.predict(test_X)
# print(pred2)

# 제출
pd.DataFrame({'pred':pred2}).to_csv('result.csv', index=False)

print(pd.read_csv('result.csv'))
