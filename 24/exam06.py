# [문제 1] 결측치 채우기
# Age 컬럼에 결측치가 있다. 이를 단순히 전체 평균으로 채우지 말고, 각 Pclass(선실 등급)별 '중앙값'으로 채우시오. 
# (예: 1등급 승객의 빈칸은 1등급 승객의 나이 중앙값으로...) 전처리가 끝난 후, Age 컬럼의 전체 평균을 구하시오. (소수점 둘째 자리 반올림)

import pandas as pd
train = pd.read_csv('csv/train.csv')

# transform('median')을 하면 각 승객의 자리에 자기 등급의 중앙값이 올라감!
n = train.groupby('Pclass')['Age'].transform('median')

train['Age'] = train['Age'].fillna(n)

Ans = train['Age'].mean()

# print(round(Ans, 2)) # 답: 29.07

# ========================================================================================================================================

# [문제 2] 스케일링 & 조건 검색
# Fare(요금) 데이터를 Min-Max Scaling (최소 0, 최대 1로 변환) 하시오. 변환된 데이터에서 값 0.5보다 큰 승객의 수를 구하시오. 
# (정수 출력) (sklearn 라이브러리 써도 되고, 수식 직접 써도 됨)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# sklearn라이브러리는 무조거 2차원을 원하므로 [[]]를 써야함
train['Fare'] = scaler.fit_transform(train[['Fare']]) 

Ans = len(train[train['Fare'] > 0.5])
print(Ans) # 답: 9

# 라이브러리 없이 공식으로 푸는 방법!!!!
# min_fare = train['Fare'].min()
# max_fare = train['Fare'].max()

# (내값 - 최소값) / (최대값 - 최소값) -> min-max 정규화 공식 외워서 풀어도 됨
# train['Fare_scaler'] = (train['Fare'] - min_fare) / (max_fare - min_fare)
# Ans = len(train[train['Fare_scaler'] > 0.5])

# ========================================================================================================================================

# [문제 3] 문자열 파생변수 & 상관관계
# Name 컬럼의 문자열 길이(글자 수)를 구해서 새로운 컬럼 Name_len을 만드시오. (공백 포함) 그리고 Name_len과 Fare(요금) 간의 
# 피어슨 상관계수(Pearson Correlation)를 구하시오. (소수점 셋째 자리에서 반올림하여 소수점 둘째 자리까지 출력. 절대값 아님)

# 문자열 길이 구할 땐 .str.len()을 사용
train['Name_len'] = train['Name'].str.len()

# 피어슨 상관계수는 corr()을 사용
# 표 형식으로 출력되기 때문에 iloc[0, 1]을 사용해 정확한 값을 지정
r = train[['Name_len', 'Fare']].corr().iloc[0, 1]

# print(round(r, 2)) # 답: 0.16

# ========================================================================================================================================

# [문제 4] 이상치 보정 (Standard Deviation 방식)
# Sex가 'female'인 승객 데이터만 추출하시오. 이들의 Fare 데이터를 기준으로, 평균으로부터 '표준편차의 1.5배'를 벗어나는 범위(Mean ± 1.5 * Std)를 이상치로 간주한다. 
# 이 이상치 데이터들의 Fare 합계를 구하시오. (소수점 버리고 정수로 출력)

c = train[train['Sex'] == 'female']['Fare']

# 평균(mean)과 표준편차(std)를 구함
m = c.mean()
s = c.std()

# 표춘편차의 1.5배를 벗어나는 범위 구하기
lower = m - 1.5 * s
upper = m + 1.5 * s

result = c[(c< lower) | (c > upper)].sum()


# print(int(result)) # 답: 5185