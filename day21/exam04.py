# 실전 모의고사

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')
test = pd.read_csv('C:/csv/test.csv')

# 2유형
# Titanic 탑승객 데이터가 주어졌을 때, 탑승객의 "생존여부(Survived)"를
# 예측하는 분류모델을 생성하시오.
# 예측 결과는 result.csv로 저장하시오

# 조건
# 사용할 컬럼은 자유롭게 선택하되, 문자형 범주 컬럼 처리 필수
# 결측치 처리도 반드시 진행
# 예측 결과는 PassengerId와 Survived로 구성해 result.csv로 저장

# print(train.info())   --> 수치형: Age(결측치o), 범주형: Cabin, Embarked
# print(test.info())    --> 수치형: Age(결측치o), 범주형: Cabin, Embarked

# 전처리
train = train.drop(columns=['Cabin', 'Fare', 'Name', 'Ticket'])
test = test.drop(columns=['Cabin', 'Fare', 'Name', 'Ticket'])

train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train['Embarked'] = le.fit_transform(train['Embarked'])
test['Embarked'] = le.transform(test['Embarked'])

# print(train.info())
# print(test.info())

# 데이터 분할
from sklearn.model_selection import train_test_split

X = train.drop(columns=['PassengerId', 'Survived'])
y = train[['PassengerId', 'Survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 모델링 및 학습
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300, max_depth=30, random_state=10)

rfc = rfc.fit(X_train, y_train)

pred1 = rfc.predict(X_test)

final_pred = rfc.predict(test.drop(columns=['PassengerId']))
print("final_pred:", final_pred[:5])


pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived' : final_pred}).to_csv('result.csv', index=False)

print(pd.read_csv('result.csv'))