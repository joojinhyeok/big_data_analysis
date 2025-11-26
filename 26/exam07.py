# [문제 3] 3유형 - 상관분석 & 회귀분석 
# Age(나이)와 Fare(요금) 간의 관계를 분석하려 한다.
# 두 변수(Age, Fare)의 피어슨 상관계수를 구하시오. (소수점 셋째 자리 반올림)
# Age를 독립변수(X), Fare를 종속변수(Y)로 하는 **단순 선형 회귀분석(ols)**을 수행했을 때, **Age 변수의 회귀계수(Coefficient)**를 구하시오. (소수점 셋째 자리 반올림)

import pandas as pd
train = pd.read_csv('csv/train.csv')

c = train[['Age', 'Fare']].corr().iloc[0, 1]
# print(c) -> 0.096

# 선형 회귀분석
from statsmodels.formula.api import ols
model = ols('Fare ~ Age', data=train).fit()

# print(model.params) -> 0.35