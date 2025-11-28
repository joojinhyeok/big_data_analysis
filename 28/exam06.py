# [3유형] 통계적 가설 검정
import pandas as pd

train = pd.read_csv('medical_train.csv')

# Q4. 독립표본 T-검정 (이분산 가정)
# smoker가 'yes'인 그룹과 'no'인 그룹 간의 bmi 평균에 차이가 있는지 검정하시오. 
# (단, bmi 결측치는 제거하고 수행)
# 두 그룹의 bmi 데이터를 추출하시오. 독립표본 T-검정을 수행하시오. (등분산 가정 안 함)
# 검정 결과 검정통계량의 절대값을 구하시오. (소수점 넷째 자리에서 반올림하여 셋째 자리까지 출력)

train = train.dropna(subset=['bmi'])

s_yes = train[train['smoker'] == 'yes']['bmi']
s_no = train[train['smoker'] == 'no']['bmi']

from scipy.stats import ttest_ind
Ans = ttest_ind(s_yes, s_no, equal_var=False)

print(abs(Ans[0])) # 답: 0.067