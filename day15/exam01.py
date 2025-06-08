# 2유형 문제 풀이

# ------------------------------------------------------------------------------------
# 1. train.csv와 test.csv 파일을 불러오시오
import pandas as pd

train = pd.read_csv('C:/csv/train.csv')
test = pd.read_csv('C:/csv/test.csv')
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# 2. 다음 열은 제거하시오: 'PassengerId', 'Name', 'Ticket', 'Cabin'
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# print(train.shape)
# print(test.shape)
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# 3. train의 결측치는 모두 제거(dropna)
train = train.dropna()
# print(train.shape)
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# 4. test의 결측치는 다음 기준으로 처리하시오:
# 수치형 컬럼: 중앙값(median)
# 범주형 컬럼(Embarked): 'S'
print(test.info()) # 수치형/범주형 확인

# 결측치는 Age, Fare만 존재하고 Embarked도 채우라했으니 해주기
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Embarked'] = test['Embarked'].fillna('S')

# print(test['Age'].isnull().sum())
# print(test['Fare'].isnull().sum())
# print(test['Embarked'].isnull().sum())
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# 5. Sex와 Embarked 컬럼을 LabelEncoder를 사용해 숫자로 변환하시오
from sklearn.preprocessing import LabelEncoder

le_sex = LabelEncoder()
le_embarked = LabelEncoder()

train['Sex'] = le_sex.fit_transform(train['Sex'])
test['Sex'] = le_sex.transform(test['Sex'])

train['Embarked'] = le_embarked.fit_transform(train['Embarked'])
test['Embarked'] = le_embarked.transform(test['Embarked'])

# print(train.head())
# print(test.head())
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# 6. 다음 3가지 모델을 학습하고 교차검증하여 정확도를 비교하시오
from sklearn.model_selection import cross_val_score

# 데이터 분리
X = train.drop('Survived', axis=1)
y = train['Survived']

# LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

# 교차검증
# base 모델: lr, dt, rf / 최종 meta 모델: LogisticRegression
for model, name in zip([lr, dt, rf], ['LogisticRegression', 'DecisionTree', 'RandomForest']):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f'{name} 평균 정확도: {scores.mean():.4f}')
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# 7. 위 모델들을 기반으로 StackingClassifier를 구성하고, test 데이터에 대해 예측하시오
from sklearn.ensemble import StackingClassifier

# 스태킹 구성
stack_model = StackingClassifier(
    estimators=[('lr', lr), ('dt', dt), ('rf', rf)],
    final_estimator=LogisticRegression()
)

# 전체 데이터로 학습
stack_model.fit(X, y)

# test 데이터 예측
pred = stack_model.predict(test)
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# 8. 최종 결과를 submission.csv로 저장하시오
# 컬럼은 PassengerId, Survived
# index=False 옵션 필수

test_original = pd.read_csv('C:/csv/test.csv')

# PassengerId + 예측 결과로 제출용 DataFrame 생성
submission = pd.DataFrame({
    'PassengerId': test_original['PassengerId'],
    'Survived': pred
})

# 저장
submission.to_csv('C:/csv/submission.csv', index=False)
# ------------------------------------------------------------------------------------