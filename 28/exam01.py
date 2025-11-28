import pandas as pd

train = pd.read_csv('car_train.csv')

# [1유형] 데이터 전처리 (2문제)
# Q1. 그룹핑 & 중앙값
# car_train.csv 데이터를 사용하시오. **brand**별로 그룹을 나누고, 각 브랜드의 **price 중앙값(median)**을 구하시오. 그중 가장 비싼 브랜드의 중앙값을 정수로 출력하시오.
# print(train.head())

Ans1 = train.groupby('brand')['price'].median()

# print(Ans1) # 답: 35181(BMW)

# ========================================================================================================================================

# Q2. 파생변수 & 필터링
# 2023년을 기준으로 차량의 나이를 계산하여 car_age 컬럼을 만드시오. (식: 2023 - year)
# car_age가 **10년 이상(>= 10)**인 차량 중, **transmission이 'Manual'**인 차량의 mileage 평균을 구하시오. (정답은 소수점 둘째 자리에서 반올림하여 첫째 자리까지 출력)
train['car_age'] = 2023 - train['year']

# print(train['car_age'])

Ans2 = train[(train['car_age'] >= 10) & (train['transmission'] == 'Manual')]['mileage'].mean()

# print(round(Ans2, 1)) # 답: 52433.5