# [1유형] 데이터 전처리 & 분석
import pandas as pd
train = pd.read_csv('csv/train.csv')

# print(train.info())

# Q1. 시계열 (월별 집계)
# 가상의 날짜 데이터를 생성하자.
# 2023년 1월 1일부터 승객 순서대로 1일씩 증가하는 Date 컬럼을 생성하시오.
# Date에서 '월(Month)' 정보를 추출하여 Month 컬럼을 만드시오.
# **4월(April)과 5월(May)**에 탑승한 승객들의 **Age 중앙값(median)**을 구하시오. (단, Age 결측치는 제거하고 계산, 정답은 소수점 첫째 자리에서 반올림하여 정수로 출력)

train['Date'] = pd.date_range(start='2023-01-01', periods=len(train))
train['month'] = train['Date'].dt.month
# print(train['month'])

d_age = train.dropna(subset=['Age'])

# print(d_age)

Ans1 = d_age[(d_age['month'] == 4) | (d_age['month'] == 5)]['Age'].median()

# print(int(Ans1)) # 답: 28

# ========================================================================================================================================

# Q2. 문자열 길이 & 조건 필터링
# Name 컬럼의 **문자열 길이(공백 포함)**를 구하여 Name_Len 컬럼을 생성하시오. Name_Len이 30 이상인 승객들의 Fare(요금) 평균을 구하시오. 
# (소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력)

train['Name_Len'] = train['Name'].str.len()
# print(train['Name_Len'])

Ans2 = train[train['Name_Len'] >= 30]['Fare'].mean()


# print(round(Ans2, 2)) # 답: 40.85

# ========================================================================================================================================

# Q3. 이상치 탐색 (Z-Score 방식)
# Fare 변수를 표준화(Standardization, 평균 0, 표준편차 1) 하시오.
# 표준화된 값이 1.5 보다 크거나(> 1.5), -1.5 보다 작은(< -1.5) 데이터를 이상치로 간주한다.
# 이상치 데이터들의 Fare 원본 값의 합계를 구하시오. 
# (정답은 소수점 버리고 정수로 출력) (힌트: scipy.stats.zscore 쓰거나 (x - mean) / std 공식 사용)

mFare = train['Fare'].mean()
sFare = train['Fare'].std()

# print(mFare, sFare)

# [핵심] Z-score 변환 (공식 대입)
# (내값 - 평균) / 표준편차
train['Fare_Z'] = (train['Fare'] - mFare) / sFare

# 이상치 필터링 (> 1.5 or < -1.5)
outliers = train[(train['Fare_Z'] > 1.5) | (train['Fare_Z'] < -1.5)]

# print(outliers)

# 원본 값(Fare) 합계 구하기 (정수 출력)
# outliers['Fare_Z'] 합계 아님! 문제에서 '원본 값' 합계라고 했음!
Ans = int(outliers['Fare'].sum())

# print(Ans) # 답: 9676 (아마도)