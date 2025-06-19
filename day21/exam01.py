# 3유형 문제 풀이

# 문제
# Titanic 데이터셋에서 남성과 여성 승객의 요금(Fare) 분포가
# 유의미하게 다른지를 분석하려고 한다.
# 다음 문항에 답하시오.

import pandas as pd
import numpy as np
from scipy import stats

train = pd.read_csv('C:/csv/train.csv')

# print(train.info())

# (1) 남성과 여성 그룹의 평균 요금을 각각 구하시오.
#     단, Fare 결측치는 제거한 후 계산하시오.
#     (소수 셋째 자리에서 반올림)
train = train.dropna(subset='Fare')

m = train[train['Sex'] == 'male']['Fare']
w = train[train['Sex'] == 'female']['Fare']

print('남성 평균 요금: ', round(m.mean(), 3))
print('여성 평균 요금: ', round(w.mean(), 3))

# (2) 두 집단의 Fare 분산이 같은지 검정하기 위해 F-검정 통계량을
#     구하시오. 
#     단, 분자의 자유도가 더 크도록 할 것.

# 두 그룹의 분산 계산
var_m = m.var()
var_w = w.var()

# 각 그룹의 자유도 계산
dof_m = len(m) - 1  # 576
dof_w = len(w) - 1  # 313

# print(dof_m, dof_w)

# 결과출력
f_stat = var_m / var_w

print('2번 문제: ', round(f_stat, 3))


# (3) 두 집단의 평균 Fare가 통계적으로 유의미하게 다른지를 독립표본
#     T-검정으로 확인하고, p-value 값을 구하시오.
#     equal_var는 F-검정 결과에 따라 설정하시오.

# F-검정 결과 보고 equal_var 설정
# 위에 구한 F-통계량 값이 1에 가까우면 -> 등분산(equal_var = True)
#           1.5이상으로 크거나 0.5 이하로 작으면 -> 등분산x(equal=False)
if f_stat > 0.5 and f_stat < 1.5:
    equal_var = True
else:
    equal_var = False

# print(equal_var)

# ttest_ind() 실행
result = stats.ttest_ind(m, w, equal_var=equal_var)

# print(result)
# pvalue=np.float64(4.230867870042998e-08)
print('3번 문제: ', round(result.pvalue, 3))