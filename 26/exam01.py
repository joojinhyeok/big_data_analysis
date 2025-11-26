import pandas as pd
train = pd.read_csv('csv/train.csv')

# [문제 1] 1유형 - 파생변수 & 생존율 차이
# Age 컬럼의 결측치를 **전체 평균(mean)**으로 채우시오.
# Age가 **10세 미만( < 10)**인 승객을 'Child', 그 외는 **'Adult'**로 구분하는 새로운 컬럼 **Age_cat**을 만드시오.
# 'Child' 그룹의 생존율과 'Adult' 그룹의 생존율 차이(절대값)를 구하시오. (단, 정답은 소수점 둘째 자리에서 반올림)
import numpy as np

train['Age'] = train['Age'].fillna(train['Age'].mean())
# np.where(조건, 참일때값, 거짓일때값)
train['Age_cat'] = np.where(train['Age'] < 10, 'Child', 'Adult')

Ans1 = abs(train[train['Age_cat'] == 'Child']['Survived'].mean() - train[train['Age_cat'] == 'Adult']['Survived'].mean())

# print(round(Ans1, 1)) # 답: 0.2

# ========================================================================================================================================

# [문제 2] 1유형 - 이상치(Outlier) 탐색 (IQR 방식)
# **Sex가 'female'**인 승객들의 Fare 데이터를 추출하시오.
# 이 데이터의 IQR (Q3 - Q1) 값을 구하고, 이상치(Outlier)의 개수를 구하시오
c = train[train['Sex'] == 'female']['Fare']

Q1 = c.quantile(0.25)
Q3 = c.quantile(0.75)
IQR = Q3 - Q1

# print(IQR) # IQR: 42.928125

min_limit = Q1 - 1.5 * IQR
max_limit = Q3 + 1.5 * IQR

Ans2 = c[(c < min_limit) | (c > max_limit)]

# print(Ans2.count()) # 답: 28
# ========================================================================================================================================

# [문제 3] 3유형 - 단일표본 T-검정 (One-Sample T-test)
# "타이타닉 1등급(Pclass=1) 승객들의 평균 운임(Fare)은 80달러라고 할 수 있는가?"를 검정하려 한다.
# Pclass가 1인 승객의 Fare 데이터를 추출하시오.
# **단일표본 T-검정(ttest_1samp)**을 수행하여 p-value를 구하시오. (기존 평균 = 80)
# 정답은 소수점 넷째 자리에서 반올림하여 셋째 자리까지 출력하시오.
t = train[train['Pclass'] == 1]['Fare']

from scipy.stats import ttest_1samp
# import scipy.stats
# print(dir(scipy.stats))

# ttest_1samp()에는 비교할 기준값도 넣어줘야함 -> 평균 = 80
Ans3 = ttest_1samp(t, 80)

# print(Ans3) # 답: 0.437

