# -------------------------------------------------------------------------------------
# 2유형 흐름
# 1. 데이터 확인
# 2. 데이터 전처리 -> 결측치를 '최빈값'이나 '평균값', '중앙값'을 넣어보면서 모델 평가 해보기
# 3. 데이터 분할
# 4. 모델링 및 학습
# 5. 성능평가
# 6. 예측 결과 제출 및 확인
# -------------------------------------------------------------------------------------
# 분류 문제(ex: 생존, 구매여부 등) - RandomForestClassifier
# 회귀 문제(ex: 총구매액, 보험료 등) - RandomForestRegressor
# -------------------------------------------------------------------------------------
# 문제
# 빅데이터분석기사 실기 체험환경 2유형 문제
import pandas as pd

train = pd.read_csv("data/customer_train.csv")
test = pd.read_csv("data/customer_test.csv")

# 1. 데이터 확인
# print(train.info()) -> 수치형: 환불금액(결측치 존재), 범주형: 주구매상품, 주구매지점	
# print(test.info())	-> 수치형: 환불금액(결측치 존재), 범주형: 주구매상품, 주구매지점	

# 2. 데이터 전처리 

# 결측치 처리 -> 결측치를 '최빈값'이나 '평균값', '중앙값'을 넣어보면서 모델 평가 해보기
train['환불금액'] = train['환불금액'].fillna(train['환불금액'].median())	# 평균값으로 결측치 처리
test['환불금액'] = test['환불금액'].fillna(test['환불금액'].median())

# 범주형 데이터 처리
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
cols = ['주구매상품', '주구매지점']

for col in cols:
	le.fit(train[col])
	train[col] = le.transform(train[col])
	test[col] = le.transform(test[col])
	
# print(train.info())
# print(test.info())

# 3. 데이터 분할
from sklearn.model_selection import train_test_split

# 독립변수 X, 종속변수 y 
# 총구매액은 종속변수에만 필요/회원ID는 고유의 값 이므로 삭제 
X = train.drop(columns=['회원ID', '총구매액'])	
y = train['총구매액']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 4. 모델링 및 학습
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rfr.fit(X_train, y_train)
pred1 = rfr.predict(X_test)
# print(pred1)

# 5. 성능평가
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_test, pred1)	# (실제값, 예측값)
# print('rmse:', rmse)

# print(dir(sklearn.metrics))

# 6. 예측 결과 제출 및 확인
# test 데이터 예측
test_X = test.drop(columns='회원ID')

pred2 = rfr.predict(test_X)
print(pred2)

# 제출
pd.DataFrame({'pred':pred2}).to_csv('result.csv', index=False)

# print(pd.read_csv('result.csv'))