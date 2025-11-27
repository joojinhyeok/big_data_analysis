import pandas as pd

train = pd.read_csv('employee_train.csv')
test = pd.read_csv('employee_test.csv')

# 문제] 직원 퇴사 여부 예측 (Classification)
# 데이터셋: employee_train.csv (학습용), employee_test.csv (평가용) 목표: 직원의 정보 데이터를 활용하여 Attrition (퇴사 여부: 1=퇴사, 0=잔류)을 예측하시오. 
# 평가지표: ROC-AUC Score

# 제출 조건:
# 파일명: result.csv
# 컬럼: EmployeeID, Attrition (두 개 컬럼 필수)
# 주의: 평가지표에 맞는 값을 제출하시오.


# 1. 데이터 유형 파악
# print(train.info()) -> MonthlyIncome 결측치 존재
# print(test.info())

# 2. 데이터 전처리
e_id = test['EmployeeID']

# 2-1 데이터 셋 분리
X_train = train.drop(['EmployeeID', 'Attrition'], axis=1)
y = train['Attrition']
X_test = test.drop(['EmployeeID'], axis=1)

# 2-2 결측치 처리
X_train['MonthlyIncome'] = X_train['MonthlyIncome'].fillna(X_train['MonthlyIncome'].mean())
X_test['MonthlyIncome'] = X_test['MonthlyIncome'].fillna(X_train['MonthlyIncome'].mean())

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
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]

# 5. 평가
# 평가지표: ROC-AUC Score
from sklearn.metrics import roc_auc_score
ras = roc_auc_score(y_val, y_pred)
# print(ras)

# n_estimators=300, max_depth=7, random_state=42 -> 0.7815233415233416

# 6. 저장 및 제출
# 제출 조건:
# 파일명: result.csv
# 컬럼: EmployeeID, Attrition (두 개 컬럼 필수)
# 주의: 평가지표에 맞는 값을 제출하시오.

Ans = model.predict_proba(X_test)[:, 1]
result = pd.DataFrame({
    'EmployeeID': e_id,
    'Attrition': Ans
})

result.to_csv('result.csv', index=False)

print(result)