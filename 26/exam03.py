# [문제 2] 2유형 - 분류 모델링 (확률 제출)
# 데이터셋: train.csv (학습용), test.csv (평가용) 목표: 승객의 **생존 여부(Survived)**를 예측하는 모델을 만들고, 
# test.csv 승객들의 **생존 확률(1일 확률)**을 예측하여 제출하시오.

# 제출 조건:
# 파일명: result.csv
# 컬럼: PassengerId, Survived (두 개 컬럼 필수)
# 평가지표: ROC-AUC

import pandas as pd

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

# 1. 데이터 유형 파악
# print(train.info()) -> Age, Cabin, Embarked 결측치 존재
# print(test.info()) -> Age, Cabin 결측치 존재
 
# 2. 데이터 전처리
p_id = test['PassengerId']

# 2-1 데이터 셋 분리
X_train = train.drop(['Name', 'Survived', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
y = train['Survived']
X_test = test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# 2-2 결측치 처리
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())

X_train['Embarked'] = X_train['Embarked'].fillna(X_train['Embarked'].mode()[0])
X_test['Embarked'] = X_test['Embarked'].fillna(X_test['Embarked'].mode()[0])

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

X_train['Embarked'] = encoder.fit_transform(X_train['Embarked'])
X_test['Embarked'] = encoder.transform(X_test['Embarked'])

# 3. 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2)

# 4. 모델링
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]

# 5. 검증
# ROC-AUC
# import sklearn.metrics
from sklearn.metrics import roc_auc_score
# print(help(roc_auc_score))

ras = roc_auc_score(y_val, y_pred)
# print(ras)

# 6. 제출
y_pred_proba = model.predict_proba(X_test)[:, 1]
result = pd.DataFrame({
    'PassengerId': p_id,
    'Survived': y_pred_proba
})

result.to_csv('result.csv', index=False)

print(result)


# 궁금한 점
# 1. predict_proba()는 언제 사용하는지?
# -> proba()는 정답일 "가능성"을 알려줌
# 결과: [[0.9, 0.1], [0.2, 0.8], ...] (넌 90% 확률로 죽고 10% 확률로 살아...)
# 용도: "ROC-AUC" 점수 구할 때, 또는 결과물이 "확률(0.7, 0.2 등)"이어야 할 때.

# 2. predict_proba()하고 나서 뒤에 [:, 1]은 어떤 의미인지
#predict_proba()를 쓰면 결과가 **표(2차원 배열)**로 나와.
# 예를 들어 모델이 "이 사람은 80% 확률로 살았다(1)"고 예측했다면, 컴퓨터는 이렇게 두 가지 숫자를 다 뱉어내.
# 0(사망)일 확률: 20% (0.2)
# 1(생존)일 확률: 80% (0.8)
# 이 표에서 우리는 **"1(생존)일 확률"**만 필요하잖아? 그래서 슬라이싱을 하는 거야.