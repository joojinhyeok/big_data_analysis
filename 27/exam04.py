# [3유형] 통계적 가설 검정 (3문제)
import pandas as pd
train = pd.read_csv('csv/train.csv')

# Q4. 정규성 검정 & 등분산 검정 (Shapiro & Levene)
# Male(남성)과 Female(여성) 승객의 Age 분포를 비교하려 한다. (결측치는 제거하고 수행)
# 남성과 여성의 Age 데이터에 대해 각각 Shapiro-Wilk 정규성 검정을 수행하고, 두 그룹 중 **p-value가 0.05 이상(정규성 만족)**인 그룹의 이름을 쓰시오. (없으면 'None')
# 두 그룹의 **등분산 검정(Levene Test)**을 수행하여 p-value를 반올림하여 소수점 넷째 자리까지 구하시오.
from scipy.stats import shapiro
# print(help(shapiro))


# subset=['Age'] : Age 컬럼에 NaN이 있는 줄만 삭제해라!
# clean_train = train.dropna(subset=['Age'])
# m_age = clean_train[clean_train['Sex'] == 'male']['Age']
# f_age = clean_train[clean_train['Sex'] == 'female']['Age']

m_age = train[train['Sex'] == 'male']['Age'].dropna()
f_age = train[train['Sex'] == 'female']['Age'].dropna()


s_m = shapiro(m_age)
s_f = shapiro(f_age)

# print("male: ", s_m) # male:  ShapiroResult(statistic=np.float64(0.974727482745374), pvalue=np.float64(4.570551172617434e-07))
# print("female: ", s_f) # female:  ShapiroResult(statistic=np.float64(0.9847880066755258), pvalue=np.float64(0.007053757389810608))

# ========================================================================================================================================

# Q5. 독립표본 T-검정 (T-test)
# Pclass가 1등급인 승객과 3등급인 승객의 Fare 평균 차이를 검정하시오.
# **독립표본 T-검정(ttest_ind)**을 수행하되, 등분산은 가정하지 않는다(equal_var=False).
# 검정통계량(t-value)의 절대값을 구하고, 소수점 셋째 자리까지 출력하시오.
from scipy.stats import ttest_ind

p1 = train[train['Pclass'] == 1]['Fare']
p3 = train[train['Pclass'] == 3]['Fare']

t_test = ttest_ind(p1, p3, equal_var=False) 
# TtestResult(statistic=np.float64(13.150240604491971), pvalue=np.float64(1.6599902021623254e-29), df=np.float64(219.28321360332842))

# print(t_test) # 답: 13.150
# ========================================================================================================================================

# Q6. 카이제곱 검정 (Chi-square)
# **Embarked(탑승 항구)**와 Survived(생존 여부) 두 변수가 서로 독립인지 검정하시오.
# 두 변수의 **교차표(Cross Table)**를 생성하시오.
# **카이제곱 검정(chi2_contingency)**을 수행하여 p-value를 구하시오. (소수점 넷째 자리 반올림)

from scipy.stats import chi2_contingency

cross_tab = pd.crosstab(train['Embarked'], train['Survived'])

Ans = chi2_contingency(cross_tab)

p_value = Ans[1]

print(round(p_value, 3)) # 답: 0.0