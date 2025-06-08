# [2유형 - 고객 구매 분석]

# 1. train_A.csv, test_A.csv 파일을 불러오시오.
import pandas as pd

train = pd.read_csv('C:/csv/train_A.csv')
test = pd.read_csv('C:/csv/test_A.csv')

print(train.info())
print(test.info())

# 2. 다음 컬럼은 분석에서 제외하시오:
# 'CustomerID', 'Name', 'PhoneNumber'
train = train.drop(['CustomerID', 'Name', 'PhoneNumber'], axis=1)
test = test.drop(['CustomerID', 'Name', 'PhoneNumber'], axis=1)

# 3. train의 결측치는 모두 제거하시오.
train = train.dropna()


# 4. test의 결측치는 다음 기준으로 처리하시오:
# 수치형 컬럼: 중앙값(median)
# 범주형 컬럼: 최빈값(mode() 사용)
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Region'] = test['Region'].fillna(test['Region'].mode()[0])
test['Income'] = test['Income'].fillna(test['Income'].median())


# 5. Gender, Region 컬럼은 LabelEncoder를 사용하여 숫자로 변환하시오
# -> train에서 fit(), test에서는 transform()만 사용할 것
from sklearn.preprocessing import LabelEncoder
le_Gender = LabelEncoder()
le_Region = LabelEncoder()

# Gender 인코딩
le_Gender.fit(pd.concat([train['Gender'], test['Gender']], axis=0))
train['Gender'] = le_Gender.transform(train['Gender'])
test['Gender'] = le_Gender.transform(test['Gender'])

# Region 인코딩
le_Region.fit(pd.concat([train['Region'], test['Region']], axis=0))
train['Region'] = le_Region.transform(train['Region'])
test['Region'] = le_Region.transform(test['Region'])

# 6. 다음 3가지 모델을 학습하고, 5-폴드 교차검증으로 정확도를 비교하시오:
# LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
# print("모델 학습 완료")

# 5-폴드 교차검증
from sklearn.model_selection import cross_val_score

# X (입력), y (정답) 분리
X = train.drop('Purchased', axis=1)
y = train['Purchased']

# 교차검증 실행
for model, name in zip([lr, dt, rf], ['LogisticRegression', 'DecisionTree', 'RandomForest']):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f'{name} 평균 정확도: {scores.mean():.4f}')


# 7. 위 모델들을 기반으로 StackingClassifier를 구성하고, test 데이터에 대해 예측하시오
from sklearn.ensemble import StackingClassifier

Stack_model = StackingClassifier(
    estimators=[('lr', lr), ('dt', dt), ('rf', rf)],
    final_estimator=LogisticRegression()
)

Stack_model.fit(X, y)

pred = Stack_model.predict(test)

# 8. 예측 결과를 submission.csv로 저장하시오
# 컬럼은 'CustomerID', 'Purchased'
# index=False 옵션 필수

test_original = pd.read_csv('C:/csv/test_A.csv')

submission = pd.DataFrame({
    'CustomerID': test_original['CustomerID'],
    'Purchased': pred
})

submission.to_csv('C:/csv/submission_2.csv', index=False)