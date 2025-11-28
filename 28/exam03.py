import pandas as pd

train = pd.read_csv('car_train.csv')

# Q4. 독립표본 T-검정 (T-test)

# transmission이 'Automatic'인 그룹과 'Manual'인 그룹 간의 mpg(연비) 평균에 차이가 있는지 검정하시오. 
# (단, mpg의 결측치는 제거하고 수행)
# 두 그룹의 mpg 데이터를 추출하시오.
# 독립표본 T-검정을 수행하시오. (등분산 가정 안 함: equal_var=False)
# 검정 결과 p-value를 소수점 넷째 자리에서 반올림하여 셋째 자리까지 출력하시오.

# 독립표본 T-검정은 ttest_ind

train = train.dropna(subset=['mpg'])

t_a = train[train['transmission'] == 'Automatic']['mpg']
t_m = train[train['transmission'] == 'Manual']['mpg']

from scipy.stats import ttest_ind

Ans = ttest_ind(t_a, t_m, equal_var = False)

# print(round(Ans[1], 3)) # 답: 0.165
