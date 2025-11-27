import pandas as pd
train= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/train.csv')
test= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/test.csv')

# display(train.head())
# display(test.head())

# 2유형!!!!!!!!
# stroke를 예측해야함(뇌졸중 여부)
# 평가지표: roc-score

# 1. 데이터 유형 파악
# print(train.info()) #-> bmi 결측치(float형)
# print(test.info())

# 2. 데이터 전처리
i = test['id']

# ====================================================================================
# "숫자로 바꿔봐! 안 되는 놈은 NaN(빈칸)으로 만들어!"
# temp = pd.to_numeric(train['age'], errors='coerce')
# temp2 = pd.to_numeric(test['age'], errors='coerce')

# "NaN이 된 놈들(원래 숫자가 아니었던 놈들)만 보여줘!"
# print(train[temp.isna()])
# print(test[temp2.isna()])
train['age'] = train['age'].astype(str).str.replace('*', '').astype(int)
test['age'] = test['age'].astype(str).str.replace('*', '').astype(int)
# ====================================================================================

# 2-1 데이터 셋 분리
X_train = train.drop(['id', 'stroke'], axis=1)
y = train['stroke']
X_test = test.drop(['id'], axis=1)

# 2-2 결측치 채우기
X_train['bmi'] = X_train['bmi'].fillna(X_train['bmi'].mean())
X_test['bmi'] = X_test['bmi'].fillna(X_train['bmi'].mean())

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

# 4. 모델 학습 및 검증
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]

# 5. 평가
# 평가지표: roc-score
# import sklearn.metrics
# print(dir(sklearn.metrics))
from sklearn.metrics import roc_auc_score
ras = roc_auc_score(y_val, y_pred)

# print(ras) # 0.8304232804232804 => 실제 시험장에서는 n_estimators랑 max_depth값 바꿔보고 결측치 채우는 곳도 바꾸면서 제일 높은 점수 제출

#6. 제출
s_pred = model.predict_proba(X_test)[:, 1]
result = pd.DataFrame({
    'id': i,
    'stroke': s_pred
})

result.to_csv('result.csv', index=False)

print(result)