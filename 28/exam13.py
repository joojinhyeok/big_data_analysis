# [2유형] 머신러닝 
import pandas as pd

train = pd.read_csv('ecommerce_train.csv')
test = pd.read_csv('ecommerce_test.csv')

# Q4. 고객 이탈 예측
# 데이터셋: ecommerce_train.csv, ecommerce_test.csv 
# 목표: 고객 정보를 이용하여 **Churn (이탈 여부: 1=이탈, 0=유지)**을 예측하시오. 
# 평가지표: F1-score

# 제출 조건:
# 파일명: result.csv
# 컬럼: CustomerID, Churn
# 주의: 평가지표(F1-score)에 맞는 예측값을 제출하시오.

# 1. 데이터 유형 파악
# print(train.info())
# print(test.info())
# print(train.head())

# 2. 데이터 전처리
c_id = test['CustomerID']
# train['JoinDate'] = pd.to_datetime(train['JoinDate'])
# test['JoinDate'] = pd.to_datetime(test['JoinDate'])

# 2-1 데이터 셋 분리
X_train = train.drop(['CustomerID', 'Churn', 'Email', 'JoinDate'], axis=1)
y = train['Churn']
X_test = test.drop(['CustomerID', 'Email', 'JoinDate'], axis=1)

# 2-2 결측치 처리
X_train['TotalSpend'] = X_train['TotalSpend'].fillna(X_train['TotalSpend'].mean())
X_test['TotalSpend'] = X_test['TotalSpend'].fillna(X_train['TotalSpend'].mean())

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
    encoder = LabelEncoder()
    X_train[col] = encoder.fit_transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col])


# 3. 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2)

# 4. 모델링
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

# 5. 평가
# 평가지표: F1-score
from sklearn.metrics import f1_score
# import sklearn.metrics
# print(dir(sklearn.metrics))

f_score = f1_score(y_val, y_pred)

# print(f_score) # 0.6751054852320675

# 6. 결과 저장 및 제출
# 제출 조건:
# 파일명: result.csv
# 컬럼: CustomerID, Churn
# 주의: 평가지표(F1-score)에 맞는 예측값을 제출하시오.

c_pred = model.predict(X_test)
result = pd.DataFrame({
    'CustomerID': c_id,
    'Churn': c_pred
})

result.to_csv('result.csv', index=False)

print(result)