# [제3유형] 통계적 가설 검정 (3문제)
import pandas as pd

train = pd.read_csv('edu_train.csv')


# Q4. 독립표본 T-검정
# **Device**가 **'PC'**인 그룹과 **'Mobile'**인 그룹 간의 Score 평균에 차이가 있는지 검정하시오.
# 독립표본 T-검정을 수행하시오. (등분산 가정 안 함)
# 검정통계량의 절대값을 구하고, 소수점 셋째 자리 미만은 버림하여 출력하시오. 
# (예: 1.23456 -> 1.23)
from scipy.stats import ttest_ind

dp = train[train['Device'] == 'PC']['Score']
dm = train[train['Device'] == 'Mobile']['Score']

Ans1 = ttest_ind(dp, dm, equal_var=False)

# print(int(abs(Ans1[0]) * 100) / 100) # 답: 0.69

# Q5. 카이제곱 검정
# **Course**와 Device 변수가 서로 독립인지 검정하시오.
# 카이제곱 검정을 수행하고, 자유도를 구하시오. 
# (정답은 정수로 출력)

from scipy.stats import chi2_contingency

ct= pd.crosstab(train['Course'], train['Device'])

Ans2 = chi2_contingency(ct)

# print(Ans2) # 답: 3


# Q6. 일원배치 분산분석
# Course의 4가지 카테고리(Python, DataScience, AI, WebDev) 간에 Score 평균의 차이가 있는지 검정하시오.
# 검정 결과 p-value를 구하고, 반올림하여 소수점 넷째 자리까지 출력하시오.

from scipy.stats import f_oneway

# import scipy.stats
# print(dir(scipy.stats))

p = train[train['Course'] == 'Python']['Score']
d = train[train['Course'] == 'DataScience']['Score']
a = train[train['Course'] == 'AI']['Score']
w = train[train['Course'] == 'WebDev']['Score']

Ans3 = f_oneway(p, d, a, w)

print(round(Ans3[1], 4)) # 답: 0.7858