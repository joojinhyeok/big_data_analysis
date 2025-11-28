# [제3유형] 통계적 가설 검정 (3문제)
import pandas as pd

train = pd.read_csv('shop_train.csv')

# Q4. 독립표본 T-검정
# **MemberType**이 **'VIP'**인 그룹과 **'Bronze'**인 그룹 간의 ReviewScore 평균에 차이가 있는지 검정하시오.
# 독립표본 T-검정을 수행하시오. (등분산 가정 안 함)
# 검정 결과 **검정통계량(statistic)**의 절대값을 구하고, 소수점 넷째 자리에서 반올림하여 셋째 자리까지 출력하시오.
m_v = train[train['MemberType'] == 'VIP']['ReviewScore']
m_b = train[train['MemberType'] == 'Bronze']['ReviewScore']

from scipy.stats import ttest_ind
Ans1 = ttest_ind(m_v, m_b, equal_var=False)

# print(round(Ans1[0], 3)) # 답: 0.486

# Q5. 정규성 검정 (Shapiro-Wilk)
# Age 컬럼 데이터가 정규분포를 따르는지 검정하시오.
# Shapiro-Wilk 검정을 수행하시오.
# p-value가 0.05보다 **크면 'Normal', 작으면 'Not Normal'**을 출력하시오.
from scipy.stats import shapiro

a = shapiro(train['Age'])

import numpy as np

Ans2 = np.where(a[1] > 0.05, 'Normal', 'Not Normal') 
 
# print(Ans2) # 답: Not Normal

# Q6. 카이제곱 검정 (독립성)
# **Category**와 Gender 변수가 서로 관련이 있는지 검정하시오.
# 교차표를 생성하시오.
# 카이제곱 검정을 수행하고, p-value를 구하시오. (소수점 넷째 자리 미만은 버림하여 넷째 자리까지 출력. 예: 0.12349 -> 0.1234)

from scipy.stats import chi2_contingency

ct = pd.crosstab(train['Category'], train['Gender'])

Ans3 = chi2_contingency(ct)
result = int(Ans3[1] * 10000) / 10000

print(result) # 답: 0.7833730516987466 -> 0.7833