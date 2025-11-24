import pandas as pd

train = pd.read_csv('csv/train.csv')
test = pd.read_csv('csv/test.csv')

# print(train.head())

# 목표: 타이타닉 생존자 예측 모델 만들기

# 제공된 train.csv로 모델을 학습시키고, test.csv 파일의 승객들이 살았을지(1), 죽었을지(0) 예측해서, 
# PassengerId,Survived 컬럼을 가진 result.csv 파일을 생성하시오.

# 데이터 유형 파악
# print(train.info())
# print(test.info())

# 1. 데이터 전처리
# (1-1) 데이터 셋 분리
X_train = train.drop(['Survived', 'Cabin', 'Ticket', 'Name'], axis=1)
y = train['Survived']
X_test = test.drop(['Cabin', 'Ticket', 'Name'], axis=1)

# (1-2): 결측치 처리
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())

X_train['Embarked'] = X_train['Embarked'].fillna(X_train['Embarked'].mode()[0]) # 최빈값 'S'로 채움
X_test['Embarked'] = X_test['Embarked'].fillna(X_test['Embarked'].mode()[0]) 

X_test['Fare'] = X_test['Fare'].fillna(X_train['Fare'].mean())

# print(X_train.isna().sum(), X_test.isna().sum()) -> 여기 왜 isna().sum()을 사용했는데 숫자로 안나오고 각 컬럼의 결측치 개수가 나오지?

# (1-3): 수치형 변수 스케일링
# 수치형 변수 스케일링 -> MinMaxScaler 고정!!
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# 수치형 변수 가져오기
num_columns = X_train.select_dtypes(exclude=object).columns

# 스케일링
X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
X_test[num_columns] = scaler.transform(X_test[num_columns])

# (1-4): 범주형 변수 인코딩
# 범주형 변수 인코딩 -> LabelEncoder 고정!!
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

X_train['Sex'] = encoder.fit_transform(X_train['Sex'])
X_test['Sex'] = encoder.transform(X_test['Sex'])

X_train['Embarked'] = encoder.fit_transform(X_train['Embarked'])
X_test['Embarked'] = encoder.transform(X_test['Embarked'])

# 2. 데이터 분리
# 데이터 분리 -> train_test_split 고정!!
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2)

# 3. 모델 학습 및 검증
# 분류 문제이므로 RandomForestClassifier를 사용!
# 회귀 문제라면 RandomForestRegressor를 사용
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# fit()학습 시에는 train 데이터만
model.fit(X_train, y_train)
# predict()로 ??할 땐 X_val 데이터만
y_val_pred = model.predict(X_val)

# 4. 평가 
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_val, y_val_pred)
# print("정확도: ", acc)

# 5. 결과 저장
y_pred = model.predict(X_test)
result = pd.DataFrame({
    'PassengerId': test['PassengerId'],  # 원본 test 데이터에서 ID 가져오기
    'Survived': y_pred                   # 예측한 정답 넣기
})
result.to_csv('result.csv', index=False)

# 확인
print(pd.read_csv('result.csv').head())