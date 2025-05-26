# Day 11 - VotingClassifier 앙상블 모델 만들기

# ----------------------------------------------------------
# 1단계 : 기본 VotingClassifier 실습습
# ----------------------------------------------------------
# (1) 필요한 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# (2) 데이터 불러오기 및 전처리
train = pd.read_csv('C:/csv/train.csv')

# (3) 결측치 처리
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

# (4) 범주형 -> 숫자형 변환
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'S':0, 'C':1, 'Q':2})

# (5) feature, target 설정
X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (6) 개별 모델 정의
Logistic_model = LogisticRegression(max_iter=1000)
Tree_model = DecisionTreeClassifier()
KN_model = KNeighborsClassifier()

# (7) Hard VotingClassifier 구성
voting_clf_soft = VotingClassifier(estimators=[
    ('lr', Logistic_model),
    ('T', Tree_model),
    ('Kn', KN_model)
], voting = 'soft') # hard는 다수의 결정 기반 <-> soft는 확률 기반

voting_clf_soft.fit(X_train, y_train)
y_pred = voting_clf_soft.predict(X_test)

acc_soft = accuracy_score(y_test, y_pred)
print(f'soft Voting Accuracy: {acc_soft:.4f}')

