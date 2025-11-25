import pandas as pd
train = pd.read_csv('csv/train.csv')

# [문제 1] 1유형 - 날짜 처리 & 필터링
# train.csv에는 탑승 날짜 정보가 없지만, 있다고 가정하고 가상의 날짜 데이터를 만들자.
# pd.to_datetime('2022-01-01')부터 시작해서 승객 순서대로 하루씩 증가하는 'Date' 컬럼을 추가해. (예: 1번 승객 1월 1일, 2번 승객 1월 2일...)
# 추가된 'Date' 컬럼을 기준으로, '주말(토, 일)'에 탑승한 승객들의 Fare(요금) 평균을 구하시오.
# 정답은 소수점 둘째 자리에서 반올림하시오.

# print(train.info())

# 날짜 데이터 만들기
# pd.date_range(start='원하는 시작 날짜', periods=??)로 만듦
train['Date'] = pd.date_range(start = '2022-01-01', periods=len(train))

train['Date_day'] = train['Date'].dt.dayofweek

# 주말인 것들만 weekend에 추가
weekend = train[train['Date_day'] >= 5]

Ans1 = weekend['Fare'].mean()

# print(round(Ans1, 1)) # 답: 28.6

# ========================================================================================================================================

# [문제 2] 1유형 - 파생변수 & 그룹핑
# Age 컬럼을 활용해 'Age_group' 컬럼을 만드시오. (0~9세: 0, 10~19세: 10, ... 70세 이상: 70 처럼 10단위 내림)
# Age_group 별로 Survived(생존율) 평균이 가장 높은 연령대를 찾고, 그 연령대의 Pclass(선실 등급) 평균을 구하시오. (소수점 버리고 정수로 출력)

train['Age_group'] = (train['Age'] // 10) * 10

a = train.groupby('Age_group')['Survived'].mean() # 80대가 가장 높음

Ans2 = train[train['Age_group'] == 80.0]['Pclass'].mean().astype(int)

# print(Ans2) # 답: 1

# ========================================================================================================================================

# [문제 3] 3유형 - 독립성 검정 (맛보기)
# Sex(성별)과 Survived(생존 여부)가 서로 관련이 있는지 **카이제곱 검정(Chi-squared test)**을 수행하시오.
# 먼저 **교차표(pd.crosstab)**를 만들고, chi2_contingency를 돌려서나온 검정통계량(statistic) 값을 출력하시오. (소수점 셋째 자리 반올림)
from scipy.stats import chi2_contingency

# 검정통계량 돌리기 위해 크로스테이블 생성
train_cross = pd.crosstab(train['Sex'], train['Survived'])

Ans3 = chi2_contingency(train_cross)

print(Ans3) # 답: 260.71702016732104