# [제1유형] 데이터 전처리 및 분석 (3문제)
import pandas as pd

train = pd.read_csv('housing_train.csv')

# Q1. 조건부 이상치 탐색
# housing_train.csv 데이터를 사용하시오. 
# LotArea 컬럼의 결측치를 중앙값으로 대체한 후, Neighborhood가 'C'인 데이터의 LotArea 표준편차를 구하시오. 
# (정답은 소수점 둘째 자리에서 반올림하여 첫째 자리까지 출력)

# print(train.info())

train['LotArea'] = train['LotArea'].fillna(train['LotArea'].median())

Ans1 = train[train['Neighborhood'] == 'C']['LotArea'].std()

# print(round(Ans1, 1)) # 답: 4944.6

# ========================================================================================================================================

# Q2. 파생변수 생성 및 그룹핑
# YearBuilt 컬럼을 활용하여 HouseAge (2023 - YearBuilt) 컬럼을 생성하시오. HouseAge가 30년 미만인 집들 중, OverallQual이 가장 높은 점수를 가진 집들의 SalePrice 평균을 구하시오. 
# (정답은 정수로 출력)

train['HouseAge'] = 2023 - train['YearBuilt']

c = train[train['HouseAge'] < 30]['OverallQual'].value_counts()

# print(c) -> 10이 가장 높은 점수

Ans2 = train[(train['OverallQual'] == 10) & (train['HouseAge'] < 30)]['SalePrice'].mean()

print(int(Ans2)) # 답: 437094

# ========================================================================================================================================

# Q3. 상위 n개 추출
# Heating 방식별로 SalePrice 평균을 구한 뒤, 평균 가격이 가장 비싼 상위 2개 난방 방식의 이름을 확인하시오. 
# 해당 2개 난방 방식을 사용하는 집들의 GrLivArea 평균을 구하시오. (정답은 소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력)

s_mean = train.groupby('Heating')['SalePrice'].mean()

# print(s_mean) -> Grav, GasW

Ans3 = train[(train['Heating'] == 'Grav') | (train['Heating'] == 'GasW')]['GrLivArea'].mean()

# print(round(Ans3, 2)) # 답: 1509.26