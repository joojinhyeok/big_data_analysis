# # [3유형] 통계적 가설 검정 (2문제)
import pandas as pd

train = pd.read_csv('ecommerce_train.csv')

# Q5. 독립표본 T-검정
# DiscountUsed가 1(사용)인 그룹과 0(미사용)인 그룹 간의 TotalSpend 평균에 차이가 있는지 검정하시오. 
# (단, TotalSpend 결측치는 제거하고 수행)
# 독립표본 T-검정을 수행하시오. (등분산 가정 안 함)
# 검정 결과 검정통계량의 절대값을 구하고, **소수점 넷째 자리 미만은 버림**하여 넷째 자리까지 출력하시오. 

train = train.dropna(subset=['TotalSpend'])

from scipy.stats import ttest_ind

d1 = train[train['DiscountUsed'] == 1]['TotalSpend']
d0 = train[train['DiscountUsed'] == 0]['TotalSpend']

Ans1 = ttest_ind(d1, d0, equal_var=False)

print(int(abs(Ans1[0]) * 10000) / 10000) # 답: 0.542
 
# Q6. 카이제곱 검정 (독립성 검정)
# **Region**과 Churn 변수가 서로 독립인지 검정하시오.
# 카이제곱 검정을 수행하고, p-value가 0.05보다 **작으면 'Yes', 크거나 같으면 'No'**를 출력하시오.

from scipy.stats import chi2_contingency
ct = pd.crosstab(train['Region'], train['Churn'])

Ans2 = chi2_contingency(ct)[1]

# print(Ans2)

import numpy as np

result = np.where(Ans2 < 0.05, 'Yes', 'No')

# print(result) # 답: No