import pandas as pd

train = pd.read_csv('car_train.csv')
test = pd.read_csv('car_test.csv')

# Q3. 중고차 가격 예측 (Regression)
# 데이터셋: car_train.csv, car_test.csv 
# 목표: 차량 정보를 이용하여 **price (가격)**를 예측하시오. 
# 평가지표: RMSE (Root Mean Squared Error)
# 제출 조건:
# 파일명: result.csv
# 컬럼: id, price
# 주의: 회귀 문제이므로 predict()를 사용해야 함!

# 1. 데이터 유형 파악
# print(train.info()) -> mpg 결측치(float형)
# print(test.info())
# print(train.head())

# 2. 데이터 전처리
c_id = test['id']

# 2-1 데이터 셋 분리
X_train = train.drop(['id', 'price'], axis=1)
y = train['price']
X_test = test.drop(['id'], axis=1)

# 2-2 결측치 처리
X_train['mpg'] = X_train['mpg'].fillna(X_train['mpg'].mean())
X_test['mpg'] = X_test['mpg'].fillna(X_train['mpg'].mean())

# 2-3 수치형 변수 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

num_columns = X_train.select_dtypes(exclude='object').columns

X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
X_test[num_columns] = scaler.transform(X_test[num_columns])

# 2-4 범주형 변수 인코딩
from sklearn.preprocessing import LabelEncoder

obj_columns = X_train.select_dtypes(include='object').columns

for col in obj_columns:
    encoder = LabelEncoder()
    X_train[col] = encoder.fit_transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col])


# 3. 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2)

# 4. 모델링 및 검증
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=300, max_depth=7, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

# 5. 평가
# 평가지표: RMSE (Root Mean Squared Error)
from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_val, y_pred)

# print(rmse) # 답: 2933.1063982409846

# 6. 저장 및 제출
# 제출 조건:
# 파일명: result.csv
# 컬럼: id, price

price_pred = model.predict(X_test)
result = pd.DataFrame({
    'id': c_id,
    'price': price_pred
})

result.to_csv('result.csv', index=False)

print(result)