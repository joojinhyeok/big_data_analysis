# [제2유형] 머신러닝 (예측 모델링)
import pandas as pd

train = pd.read_csv('housing_train.csv')
test = pd.read_csv('housing_test.csv')

# Q4. 주택 가격 예측

# 데이터셋: housing_train.csv, housing_test.csv 
# 목표: 주택 정보를 이용하여 SalePrice를 예측하시오. 
# 평가지표: RMSE (Root Mean Squared Error)

# 제출 조건:
# 파일명: result.csv
# 컬럼: HouseID, SalePrice (컬럼명 준수)
# 평가지표에 맞는 모델링 및 결과 도출

# 1. 데이터 유형 파악
# print(train.info())
# print(test.info())   -> LotArea 결측치 존재
# print(train.head())

# 2. 데이터 전처리
h_id = test['HouseID']

# 2-1 데이터 셋 분리
X_train = train.drop(['HouseID', 'SalePrice'], axis=1)
y = train['SalePrice']
X_test = test.drop(['HouseID'], axis=1)

# 2-2 결측치 처리
X_train['LotArea'] = X_train['LotArea'].fillna(X_train['LotArea'].mean())
X_test['LotArea'] = X_test['LotArea'].fillna(X_train['LotArea'].mean())

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

# 4. 모델링
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, max_depth=9, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

# 5. 평가
# 평가지표: RMSE (Root Mean Squared Error)
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_val, y_pred)

# print(rmse) # 8624.434946824542 -> 너무 높게 나온거같은데 괜찮나

# 6. 저장 및 제출
# 제출 조건:
# 파일명: result.csv
# 컬럼: HouseID, SalePrice (컬럼명 준수)
# 평가지표에 맞는 모델링 및 결과 도출

sp_pred = model.predict(X_test)
result = pd.DataFrame({
    'HouseId': h_id,
    'SalePrice': sp_pred
})

result.to_csv('result.csv', index=False)

print(result)