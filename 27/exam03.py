# [1유형] 데이터 전처리 & 분석 (3문제)
import pandas as pd
train = pd.read_csv('csv/train.csv')

# Q1. 시계열 (날짜 변환 & 요일)
# 가상의 시계열 데이터를 만들어보자.
# 2023년 1월 1일부터 시작하는 Date 컬럼을 생성하시오. (승객 순서대로 1일씩 증가)
# Date 컬럼을 datetime 형으로 변환하고,
# **'금요일(Friday)'**에 탑승한 승객들 중 **Survived가 1(생존)**인 승객의 수를 구하시오. (정답은 정수로 출력)

train['Date'] = pd.date_range(start='2023-01-01', periods=len(train))

train['Date'] = pd.to_datetime(train['Date'])

train['day'] = train['Date'].dt.dayofweek

Ans = train[(train['day'] == 4) & (train['Survived'] == 1)]

# print(len(Ans)) # 답: 54

# ========================================================================================================================================

# Q2. 문자열 처리 (정규표현식)
# Name 컬럼에서 호칭을 분석하려 한다.
# 호칭이 **'Mr.' (점 포함)**인 그룹과 **'Mrs.' (점 포함)**인 그룹을 각각 추출하시오.
# 두 그룹의 Fare(요금) 평균의 합을 구하시오. (단, 소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력)

mr = train[train['Name'].str.contains('Mr\.')]
mrs = train[train['Name'].str.contains('Mrs\.')]

Ans = mr['Fare'].mean() + mrs['Fare'].mean()

# print(round(Ans, 2)) # 답: 69.58

# ========================================================================================================================================

# Q3. 스케일링 & 조건 필터링
# Fare 데이터를 Min-Max Scaling (0~1 사이로 변환) 하시오.
# 변환된 Fare 값이 0.5 보다 큰(> 0.5) 승객들의 수를 구하시오.
# (라이브러리 MinMaxScaler를 써도 되고, 공식을 써도 됨) (정답은 정수로 출력)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train['Fare'] = scaler.fit_transform(train[['Fare']])

Ans = len(train[train['Fare'] > 0.5])

# print(Ans) # 답: 9명