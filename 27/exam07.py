import pandas as pd

train = pd.read_csv('csv/train.csv')

# [문제 1] 정규성 검정 (Shapiro-Wilk)
# Pclass가 1등급인 승객과 3등급인 승객의 Age 데이터를 각각 추출하시오. (결측치는 제거)
# 두 그룹의 데이터에 대해 Shapiro-Wilk 정규성 검정을 수행하시오.
# 두 그룹의 검정 결과 중 더 작은 p-value를 구하시오. (소수점 넷째 자리에서 반올림하여 셋째 자리까지 출력)

train = train.dropna(subset=['Age'])

p1 = train[train['Pclass'] == 1]['Age']
p3 = train[train['Pclass'] == 3]['Age']

from scipy.stats import shapiro

s_p1 = shapiro(p1)
s_p3 = shapiro(p3)
# print(s_p1, s_p3)
# ShapiroResult(statistic=np.float64(0.9916934744726017), pvalue=np.float64(0.3642557817048129)) 
# ShapiroResult(statistic=np.float64(0.9734377844740231), pvalue=np.float64(4.185906522280519e-06))

# print(round(s_p1[1], 3)) # 답: 0.364

# ========================================================================================================================================

# [문제 2] 카이제곱 검정 (Chi-square)
# **Sex(성별)**와 Survived(생존 여부) 변수 간에 연관성이 있는지(독립적이지 않은지) 검정하려 한다.
# 두 변수의 **교차표(Cross Table)**를 생성하시오.
# **카이제곱 검정(chi2_contingency)**을 수행하여 **검정통계량(statistic)**을 구하시오. (소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력)

from scipy.stats import chi2_contingency

crosstab = pd.crosstab(train['Sex'], train['Survived'])

Ans2 = chi2_contingency(crosstab)

# print(Ans2)
# Chi2ContingencyResult(statistic=np.float64(205.02582752855906), pvalue=np.float64(1.6716678441395299e-46), dof=1, 
# expected_freq=array([[154.99159664, 106.00840336], [269.00840336, 183.99159664]]))

# print(round(Ans2[0], 2)) # 답: 205.03

#  ========================================================================================================================================

# [문제 3] 독립표본 T-검정 (T-test)
# **생존한 승객(Survived=1)**과 사망한 승객(Survived=0) 간의 Fare(요금) 평균에 차이가 있는지 검정하시오.
# 두 그룹의 Fare 데이터를 추출하시오.
# **독립표본 T-검정(ttest_ind)**을 수행하시오. (등분산은 가정하지 않음: equal_var=False)
# 검정 결과 p-value를 구하시오. (소수점 넷째 자리 미만은 버리고 넷째 자리까지 출력. 예: 0.12345 -> 0.1234)
from scipy.stats import ttest_ind

s1_f = train[train['Survived'] == 1]['Fare']
s0_f = train[train['Survived'] == 0]['Fare']

Ans3 = ttest_ind(s1_f, s0_f, equal_var=False)

# print(Ans3) -> TtestResult(statistic=np.float64(6.547691851114033), pvalue=np.float64(1.9646132697636508e-10), df=np.float64(368.4506883826716))

p_val = Ans3[1]

# 살리고 싶은 자릿수만큼 0의 개수를 맞추면 돼
# 소수점 2째 자리 버림: int(값 * 100) / 100
# 소수점 3째 자리 버림: int(값 * 1000) / 1000
# 소수점 4째 자리 버림: int(값 * 10000) / 10000
result = int(p_val * 10000) / 10000

# print(result) # 답: 0.0