import pandas as pd

train = pd.read_csv('csv/train.csv')

# [1유형] 데이터 전처리 & 분석 (2문제)

# Q1. 시계열 (주말 vs 평일 요금 비교)
# 가상의 날짜를 생성하여 분석해보자.
# 2023년 1월 1일부터 승객 순서대로 1일씩 증가하는 Date 컬럼을 생성하시오.
# Date를 기준으로 주말(토, 일) 그룹과 평일(월~금) 그룹으로 나누시오.
# 두 그룹의 **Fare(요금) 평균의 차이(절대값)**를 구하시오. (정답은 소수점 둘째 자리에서 반올림하여 첫째 자리까지 출력)
train['Date'] = pd.date_range(start='2023-01-01', periods=len(train))

train['Date'] = train['Date'].dt.dayofweek

weekend = train[train['Date'] > 4]
weekdays = train[train['Date'] <= 4]

Ans1 = abs(weekend['Fare'].mean() - weekdays['Fare'].mean())

# print(round(Ans1, 1)) # 답: 0.2

# ========================================================================================================================================

# Q2. 이상치 탐색 (IQR & 조건부 합계)
# Age 컬럼에 대해 IQR 방식으로 이상치를 찾으려 한다. (결측치는 제거하고 수행)
# Age의 IQR (Q3 - Q1) 값을 구하시오.
# Q1 - 1.5 * IQR 보다 작거나, Q3 + 1.5 * IQR 보다 큰 값을 이상치로 정의한다.
# 이상치에 해당하는 승객들 중 **Sex가 'female'**인 승객들의 Age 합계를 구하시오. (정답은 정수로 출력)
train = train.dropna(subset=['Age'])

Q1 = train['Age'].quantile(0.25)
Q3 = train['Age'].quantile(0.75)
IQR = Q3 - Q1

b = Q1 - 1.5 * IQR
t = Q3 + 1.5 * IQR

c = train[(train['Age'] > t) | (train['Age'] < b)]

Ans2 = c[c['Sex'] == 'female']['Age']

# print(Ans2) # 답: 0

# ========================================================================================================================================

# [3유형] 통계적 가설 검정 (2문제)

# Q3. 대응표본 T-검정 (Paired T-test)
# (가상 상황) 승객들에게 다이어트 약을 복용시켰더니 몸무게가 변했다고 가정하자.
# 기존 Fare 데이터를 **'복용 전 몸무게'**로 가정한다.
# Fare에 0.9를 곱한 값을 **'복용 후 몸무게'**로 가정한다.
# 두 변수(전, 후) 간의 평균 차이가 유의미한지 **대응표본 T-검정(ttest_rel)**을 수행하시오. (양측 검정)
# 검정 결과 p-value를 소수점 넷째 자리에서 반올림하여 셋째 자리까지 출력하시오.
from scipy.stats import ttest_rel

bf = train['Fare']
af = train['Fare'] * 0.9

Ans3 = ttest_rel(bf, af)

# print(Ans3) # TtestResult(statistic=np.float64(17.518578517300593), pvalue=np.float64(2.042532824669546e-57), df=np.int64(713))

# print(round(Ans3[1], 3)) # 답 0.0

# ========================================================================================================================================

# Q4. 카이제곱 검정 (Chi-square)
# **Pclass(선실 등급)**와 Survived(생존 여부) 두 변수가 서로 독립인지 검정하시오.
# 두 변수의 **교차표(Cross Table)**를 생성하시오.
# **카이제곱 검정(chi2_contingency)**을 수행하고, **자유도(dof)**를 구하시오. (정답은 정수로 출력)
from scipy.stats import chi2_contingency

ct = pd.crosstab(train['Pclass'], train['Survived'])

Ans4 = chi2_contingency(ct)

# print(Ans4)
# Chi2ContingencyResult(statistic=np.float64(92.90141721143321), pvalue=np.float64(6.709861749756909e-21), dof=2, 
# expected_freq=array([[110.45378151,  75.54621849], [102.73389356,  70.26610644], [210.81232493, 144.18767507]]))

# 답: 2