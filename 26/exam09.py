import pandas as pd
train = pd.read_csv('csv/train.csv')

# [문제 1-1] 그룹핑 & 상위 n개 추출
# Embarked별로 그룹을 나누고, 각 항구에서 Fare가 가장 비싼 상위 3명의 평균 요금을 구하시오. 
# 그중 가장 높은 평균 요금을 가진 항구의 평균값을 출력하시오.
#  (단, 소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력)

# 항구별 요금 상위 3개 뽑기 (Series로 나옴)
top3_series = train.groupby('Embarked')['Fare'].nlargest(3)

# 뽑힌 애들 가지고 다시 평균 구하기
# level=0은 '항구(Embarked)' 인덱스를 기준으로 묶으라는 뜻!
mean_values = top3_series.groupby('Embarked').mean()

Ans = round(mean_values, 2)
print(Ans) # 답: 512.33(C 항구)

# ========================================================================================================================================

# [문제 1-2] 시계열 데이처 처리 (가상 날짜)
# train.csv 데이터에 시계열 정보가 없으므로 가상의 날짜를 생성한다.
# 2023년 1월 1일부터 시작하는 날짜 컬럼 Date를 생성하시오. (승객 순서대로 하루씩 증가)
# Date가 **5월(May)**이면서 6월(June)이 아닌 데이터 중, Pclass가 1등급인 승객의 수를 구하시오. (정답은 정수로 출력)
train['Date'] = pd.date_range(start='2023-01-01', periods=len(train))

train['Date'] = train['Date'].dt.month

d = train[(train['Date'] == 5) & (train['Date'] != 6)]['Pclass']

Ans2 = d[d == 1]

# print(len(Ans2)) # 답: 24

# ========================================================================================================================================

# [문제 1-3] 이상치 탐색 (ESD 방식)
# Age 컬럼에 대해 ESD(Extreme Studentized Deviate) 방식으로 이상치를 찾으려 한다.
# Age의 **평균(Mean)**과 **표준편차(Std)**를 구하시오.
# 평균 - (3 * 표준편차) 보다 작거나, 평균 + (3 * 표준편차) 보다 큰 값을 이상치로 규정한다.
# 이상치에 해당하는 승객들의 Age 평균을 구하시오. (단, 결측치는 제거하고 계산하며, 정답은 소수점 버리고 정수로 출력)

train['Age'] = train['Age'].dropna()

a_mean = train['Age'].mean()
a_std = train['Age'].std()

c1 = a_mean - (3 * a_std)
c2 = a_mean + (3 * a_std)

Ans3 = train[(train['Age'] < c1) | (train['Age'] > c2)]

# print(int(Ans3['Age'].mean()))