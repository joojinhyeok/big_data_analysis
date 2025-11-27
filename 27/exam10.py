import pandas as pd
train = pd.read_csv('csv/train.csv')

# [1유형] 데이터 전처리 (2문제)

# Q1. 시계열 데이터 처리
# train.csv에 가상의 날짜를 부여한다.
# 2023년 1월 1일부터 승객 순서대로 1일씩 증가하는 Date 컬럼을 생성하시오.
# Date 컬럼을 기준으로 **'월요일(Monday)'**에 탑승한 승객들 중, **Sex가 'male'(남성)**인 승객의 수를 구하시오. (정답은 정수로 출력)
train['Date'] = pd.date_range(start='2023-01-01', periods=len(train))

train['Date'] = train['Date'].dt.dayofweek

Ans1 = train[(train['Date'] == 0) & (train['Sex'] == 'male')]

# print(int(len(Ans1))) # 답: 85

# ========================================================================================================================================

# Q2. 이상치 탐색 및 평균 계산
# Age 컬럼의 결측치를 모두 제거한 후 수행하시오.
# Age 데이터의 IQR (Q3 - Q1) 값을 구하시오.
# Q3 + 1.5 * IQR 보다 큰 값을 이상치로 간주한다.
# 이상치에 해당하는 승객들의 Fare(요금) 평균을 구하시오. (정답은 소수점 첫째 자리에서 반올림하여 정수로 출력)
train = train.dropna(subset=['Age'])

Q1 = train['Age'].quantile(0.25)
Q3 = train['Age'].quantile(0.75)
IQR = Q3 - Q1

Ans2 = train[train['Age'] > (Q3 + (1.5 * IQR))]['Fare'].mean()

# print(round(Ans2, 0)) # 답: 29

# ========================================================================================================================================

# [3유형] 통계적 가설 검정 (2문제)

# Q3. 상관분석 (Correlation)
# **Age**와 SibSp(형제자매 수) 간의 **피어슨 상관계수(Pearson Correlation)**를 구하시오. 
# (단, 결측치가 있는 행은 모두 제거하고 수행할 것) (정답은 소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력)

Ans3 = train[['Age', 'SibSp']].corr().iloc[0, 1]

# print(round(Ans3, 2)) # 답: -0.31

# ========================================================================================================================================

# Q4. 독립표본 T-검정 (T-test)

# **Sex**가 **'male'**인 그룹과 **'female'**인 그룹 간의 Fare 평균에 차이가 있는지 검정하시오.
# 두 그룹의 Fare 데이터를 추출하시오.
# 독립표본 T-검정을 수행하시오. (등분산은 가정하지 않음)
# 검정 결과 p-value를 구하시오. (정답은 소수점 다섯째 자리에서 반올림하여 넷째 자리까지 출력)

# 제일 자주 출제
from scipy.stats import ttest_ind

m = train[train['Sex'] == 'male']['Fare']
f = train[train['Sex'] == 'female']['Fare']

Ans4 = ttest_ind(m, f, equal_var=False)

# print(Ans4) # TtestResult(statistic=np.float64(-1.783952465263041), pvalue=np.float64(0.0761350357585084), df=np.float64(177.9947230971782))

# print(round(Ans4[1], 4)) # 답: 0.0761