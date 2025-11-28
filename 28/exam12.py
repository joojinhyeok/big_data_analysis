# [1유형] 데이터 전처리 (3문제)
import pandas as pd

train = pd.read_csv('ecommerce_train.csv')

# Q1. 그룹별 순위 & 합계
# Region(지역)별로 TotalSpend(총 지출액)가 가장 많은 상위 10% 고객들을 선정하려 한다.
# 각 지역(Region) 내에서 TotalSpend 기준 상위 10% 고객을 필터링하시오. (결측치는 평균으로 대체 후 수행)
# 선정된 고객들의 NumPurchases(구매 횟수) 합계를 구하시오. (정답은 정수로 출력)
# print(train.info())

train['TotalSpend'] = train['TotalSpend'].fillna(train['TotalSpend'].mean())

# my_rank 컬럼을 Region별 TotalSpend의 rank로 내림차순으로 만듦 -> 지역 내 등수를 의미
train['my_rank'] = train.groupby('Region')['TotalSpend'].rank(method='first', ascending=False)

# total_count 컬럼을 Region별 TotalSpend에 count를 적용해서 채워넣음 -> 지역 별 인구수를 의미
train['total_count'] = train.groupby('Region')['TotalSpend'].transform('count')

# print(train[['my_rank', 'total_count']])

# my_rank가 10%에 들어가는 것들만 필터링
top10_per_region = train[train['my_rank'] <= train['total_count'] * 0.1]

Ans1 = top10_per_region['NumPurchases'].sum()

# print(int(Ans1)) # 답: 3600


# Q2. 문자열 분해 & 빈도 (value_counts 응용)
# Email 컬럼에서 이메일 도메인(예: gmail.com)만 추출하시오. 가장 많이 사용되는 도메인 상위 2개에 해당하는 고객들의 Age 평균을 구하시오. 
# (정답은 소수점 둘째 자리에서 반올림하여 첫째 자리까지 출력)

# print(train['Email'])

train['E_domain'] = train['Email'].str.split("@").str[1]

# print(train['E_domain'])

d = train['E_domain'].value_counts()

# print(d) -> daum.net과 naver.com

Ans2 = train[(train['E_domain'] == 'daum.net') | (train['E_domain'] == 'naver.com')]['Age'].mean()

# print(round(Ans2, 1)) # 답: 43.6

# Q3. 날짜 변환 & 조건 필터링
# JoinDate를 활용하여 2021년에 가입했고, Satisfaction이 4점 이상인 고객 수를 구하시오. (정답은 정수로 출력)

# print(train.info())

train['JoinDate'] = pd.to_datetime(train['JoinDate'])

train['year'] = train['JoinDate'].dt.year

Ans3 = train[(train['year'] == 2021) & (train['Satisfaction'] >= 4)]

# print(len(Ans3)) # 답: 182