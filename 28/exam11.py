# [제3유형] 통계적 가설 검정 (2문제)
import pandas as pd

train = pd.read_csv('housing_train.csv')

# Q5. 독립성 검정
# CentralAir와 Neighborhood 변수가 서로 관련이 있는지(독립적이지 않은지) 검정하시오
# 카이제곱 검정을 수행하고, 검정 결과 p-value를 구하시오. (소수점 넷째 자리에서 반올림하여 셋째 자리까지 출력)

# 카이제곱 검정 -> chi2_contingency
from scipy.stats import chi2_contingency

# print(help(chi2_contingency))

ct = pd.crosstab(train['CentralAir'], train['Neighborhood'])

Ans1 = chi2_contingency(ct)

# print(round(Ans1[1], 3)) # 답: 0.657

# Q6. 상관분석
# OverallQual과 SalePrice의 피어슨 상관계수를 구하시오.
# 상관계수의 절대값이 0.7 이상이면 'Strong', 미만이면 'Weak'를 출력하시오. (단, 결측치는 제거하고 수행)
import numpy as np

Ans6 = train[['OverallQual', 'SalePrice']].corr().iloc[0, 1]

# print(Ans6)

result = np.where(abs(Ans6) >= 0.7, 'Strong', 'Weak')

# print(result) # 답: Strong