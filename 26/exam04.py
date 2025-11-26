# [문제 3] 3유형 - 일원배치 분산분석 (ANOVA)
# **Pclass(선실 등급)**가 1등급, 2등급, 3등급인 승객들 간의 Fare(요금) 평균에 유의미한 차이가 있는지 검정하시오.
# 3개 그룹(Pclass=1, 2, 3)의 Fare 데이터를 각각 추출하시오.
# **ANOVA 검정(f_oneway)**을 수행하여 **검정통계량(F-value)**을 구하시오.
# 정답은 소수점 둘째 자리에서 반올림하여 소수점 첫째 자리까지 출력하시오.

import pandas as pd
train = pd.read_csv('csv/train.csv')

P1 = train[train['Pclass'] == 1]['Fare']
P2 = train[train['Pclass'] == 2]['Fare']
P3 = train[train['Pclass'] == 3]['Fare']

from scipy.stats import f_oneway

Ans = f_oneway(P1, P2, P3)

print(Ans) # 답: 242.3