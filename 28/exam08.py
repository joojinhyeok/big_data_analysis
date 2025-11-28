import pandas as pd

train = pd.read_csv('csv/train.csv')

# Q4. 독립성 검정 (Chi-square)
# Pclass(객실 등급)와 Survived(생존 여부)의 독립성을 검정하시오.
# 두 변수의 **교차표(Cross Table)**를 생성하시오.
# 카이제곱 검정을 수행하고 검정통계량을 구하시오. (소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력)

# 카이제곱 검정 -> chi2_contingency

c_tab = pd.crosstab(train['Pclass'], train['Survived'])

from scipy.stats import chi2_contingency
Ans1 = chi2_contingency(c_tab)

# print(round(Ans1[0], 2)) # 답: 102.89

# ========================================================================================================================================

# Q5. 독립표본 T-검정
# **Embarked**가 **'S'**인 그룹과 **'C'**인 그룹의 Fare 평균 차이를 검정하시오.
# 두 그룹의 Fare 데이터를 추출하시오. (결측치 제거)
# 독립표본 T-검정을 수행하시오. (등분산 가정 안 함)
# 검정 결과 p-value가 0.05보다 작으면 'Yes', 크면 'No'를 출력하시오.

# 독립표본 T-검정 -> ttest_ind

e_s = train[train['Embarked'] == 'S']['Fare']
e_c = train[train['Embarked'] == 'C']['Fare']

from scipy.stats import ttest_ind
Ans2 = ttest_ind(e_s, e_c, equal_var=False)

import numpy as np

result = np.where(Ans2[1] < 0.05, 'Yes', 'No')

# print(result) # 답: Yes

# ========================================================================================================================================

# Q6. 상관분석 (Correlation)
# **Age**와 **Fare**의 피어슨 상관계수를 구하시오. (결측치 제거) 상관계수의 절대값이 0.3 이상이면 'High', 미만이면 'Low'를 출력하시오.

train = train.dropna(subset=['Age', 'Fare'])

cor = train[['Age', 'Fare']].corr().iloc[0, 1]

Ans3 = np.where(abs(cor) >= 0.3 , 'High', 'Low')

# print(Ans3) # 답: Low
