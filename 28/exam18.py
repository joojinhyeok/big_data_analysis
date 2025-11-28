import pandas as pd

train = pd.read_csv('shop_train.csv')

# Q1. 시계열 (시간 추출 & 필터링)
# Date 컬럼을 활용하여 오전 10시부터 오후 2시 사이에 발생한 주문 건수를 구하시오. 
# 그중 **MemberType이 'VIP'**인 주문의 비율(%)을 구하시오. 
# (비율 = 조건 만족 수 / 시간대 전체 주문 수) (정답은 소수점 둘째 자리에서 반올림하여 첫째 자리까지 출력)
# print(train.info())

train['Date'] = pd.to_datetime(train['Date'])

train['hour'] = train['Date'].dt.hour

Ans1 = len(train[(train['hour'] >= 10) & (train['hour'] <= 14) & (train['MemberType'] == 'VIP')]) / len(train[(train['hour'] > 10) & (train['hour'] < 14)])

# print(round(Ans1, 1)) # 답: 0.3


# Q2. 그룹별 순위 (최대값 추출)
# Category별로 Price가 가장 비싼 데이터를 추출하려 한다. 
# (결측치는 제거하고 수행) 
# 각 카테고리에서 가장 비싼(Max) 주문 건들의 Qty(수량) 합계를 구하시오. (정답은 정수로 출력)

train = train.dropna(subset=['Price'])

train['max_p'] = train.groupby('Category')['Price'].rank(method='first', ascending=False)


Ans2 = train[train['max_p'] == 1]['Qty'].sum()

# print(Ans2) # 답: 15

# Q3. 조건부 파생변수 & 상관분석 맛보기
# TotalAmt (Price * Qty) 컬럼을 생성하시오. 
# (결측치는 Price의 중앙값으로 대체 후 계산) Gender가 'M'인 그룹과 'F'인 그룹 각각의 TotalAmt 평균의 차이(절대값)를 구하시오. 
# (정답은 정수로 출력 - 소수점 버림)

train['Price'] = train['Price'].fillna(train['Price'].median())

train['TotalAmt'] = train['Price'] * train['Qty']

m_total = train[train['Gender'] == 'M']['TotalAmt'].mean()
f_total = train[train['Gender'] == 'F']['TotalAmt'].mean()

Ans3 = int(abs(m_total - f_total))

# print(Ans3) # 답: 76510