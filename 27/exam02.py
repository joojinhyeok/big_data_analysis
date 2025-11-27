import pandas as pd

train = pd.read_csv('csv/train.csv')

# print(train.info())

# [문제 1-1] 그룹핑 & 최소값(nsmallest)
# Pclass별로 그룹을 나누고, 각 등급에서 Age가 가장 적은 하위 5명의 평균 나이를 구하시오. 
# 그중 가장 낮은 평균 나이를 가진 등급의 평균값을 출력하시오. 
# (단, 결측치는 제거하고 계산하며, 정답은 소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력)

# 결측치 제거
train['Pclass'] = train['Pclass'].dropna()
# print(train['Pclass'].isna().sum())

# 나이 순으로 정렬
a_sort = train.sort_values('Age')

# 어린순으로 정렬했으니 Pclass별로 상위 5명을 뽑아 새로운 DataFrame으로 만듦
p_group = a_sort.groupby('Pclass').head()

Ans = p_group.groupby('Pclass')['Age'].mean()

# print(round(Ans, 2)) # 답: 1: 6.38, 2: 0.87, 3: 0.78
# ========================================================================================================================================

# [문제 1-2] 시계열 데이처 처리 (요일 필터링)
# 가상의 날짜를 생성해보자.
# 2023년 1월 1일부터 시작하는 날짜 컬럼 **Date**를 생성하시오. (승객 순서대로 하루씩 증가)
# Date 컬럼을 기준으로, **'일요일(Sunday)'**에 탑승한 승객들 중 **Sex가 'male'(남성)**인 사람의 수를 구하시오. 
# (정답은 정수로 출력)

train['Date'] = pd.date_range(start='2023-01-01', periods=len(train))

train['Date'] = train['Date'].dt.dayofweek

Ans = train[(train['Date'] == 6) & (train['Sex'] == 'male')]

# print(len(Ans)) # 답: 91

# ========================================================================================================================================

# [문제 1-3] 이상치 탐색 (IQR 응용)
# Fare(요금) 컬럼에 대해 IQR 방식으로 이상치를 찾으려 한다.
# Fare의 IQR (Q3 - Q1) 값을 구하시오.
# Q3 + 1.5 * IQR 보다 큰 값을 '큰 이상치(Max Outlier)'라고 정의한다. (작은 이상치는 무시)
# '큰 이상치'에 해당하는 승객들의 Fare 평균을 구하시오. 
# (정답은 소수점 버리고 정수로 출력)

Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1

top_out = Q3 + (IQR * 1.5)

Ans = train[train['Fare'] > top_out]['Fare'].mean().astype(int)

# print(Ans) # 답: 128