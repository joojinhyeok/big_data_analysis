# [문제] 자전거 대여량(count)을 예측하라!
# 데이터셋: bike_train.csv, bike_test.csv 목표: 주어진 날씨, 날짜 정보를 이용하여 count(자전거 대여량)를 예측하시오. 
# 평가지표: RMSE (Root Mean Squared Error)
# 제출 조건:
# 파일명: result.csv
# 컬럼: datetime, count (컬럼명 준수)

import pandas as pd

train = pd.read_csv('bike_train.csv')
test = pd.read_csv('bike_test.csv')


# 1. 데이터 유형 파악
# print(train.info())
# print(test.info())
# print(train.head())

# 2. 데이터 전처리
dtime = test['datetime']

# 2-1 데이터 셋 분리
X_train = train.drop(['datetime', 'count'], axis=1)
y = train['count']
X_test = test.drop(['datetime'], axis=1)

# 2-2 결측치 처리
X_train['humidity'] = X_train['humidity'].fillna(X_train['humidity'].mean())
X_test['humidity'] = X_test['humidity'].fillna(X_train['humidity'].mean())

# 2-3 수치형 변수 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

num_columns = X_train.select_dtypes(exclude=['object', 'datetime']).columns

X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
X_test[num_columns] = scaler.transform(X_test[num_columns])

# 2-4 범주형 변수 인코딩
from sklearn.preprocessing import LabelEncoder

obj_columns = X_train.select_dtypes(include='object').columns

for col in obj_columns:
    encoder =LabelEncoder()
    X_train[col] = encoder.fit_transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col])

# 3. 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2)

# 4. 모델링
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=300, max_depth=7, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

# 5. 평가
# 평가지표: RMSE (Root Mean Squared Error)
# import sklearn.metrics 
# print(dir(sklearn.metrics))
from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_val, y_pred)
# print(rmse) -> 낮을 수록 좋음. 22.114915030424008

# 6. 결과 제출
# 제출 조건:
# 파일명: result.csv
# 컬럼: datetime, count (컬럼명 준수)
count_pred = model.predict(X_test)
result = pd.DataFrame({
    'datetime': dtime,
    'count': count_pred
})

result.to_csv('result.csv', index=False)

print(result)