# [제1유형] 데이터 핸들링 (3문제)
import pandas as pd

train = pd.read_csv('job_train.csv')

# Q1. 그룹별 중앙값 & 차이
# Major(전공)별로 그룹을 나누어 Toeic 점수의 중앙값을 구하시오. 
# 가장 높은 중앙값을 가진 전공과 가장 낮은 중앙값을 가진 전공의 점수 차이(절대값)를 구하시오. (정답은 정수로 출력)
# Ans1 = train.groupby('Major')['Toeic'].median()

# print(Ans1) # -> CS / Biz 

result1 = train[train['Major'] == 'CS']['Toeic'].median() -train[train['Major'] == 'Biz']['Toeic'].median()

# print(int(result1)) # 답: 9

# Q2. 시계열 & 필터링
# **JoinDate**를 활용하여 **'5월(May)'**에 가입한 지원자 중, Internship 경력이 있는(1) 사람의 비율(%)을 구하시오. 
# (비율 = 조건 만족 수 / 5월 전체 지원자 수) (정답은 소수점 둘째 자리에서 반올림하여 첫째 자리까지 출력)

# print(train.info())
train['JoinDate'] = pd.to_datetime(train['JoinDate'])

train['month'] = train['JoinDate'].dt.month

Ans2 = len(train[(train['month'] == 5) & (train['Internship'] == 1)]) / len(train[train['month'] == 5])

# print(round(Ans2, 1)) # 답: 0.3

# Q3. 이상치 탐색 
# GPA 컬럼의 결측치를 평균으로 대체한 후, IQR 방식으로 이상치를 찾으시오. 
# Q1 - 1.5*IQR 미만이거나 Q3 + 1.5*IQR 초과인 데이터를 이상치로 간주한다. 
# 이상치 데이터들의 GPA 평균을 구하시오. (정답은 소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력)

train['GPA'] = train['GPA'].fillna(train['GPA'].mean())

Q1 = train['GPA'].quantile(0.25)
Q3 = train['GPA'].quantile(0.75)
IQR = Q3 - Q1

b = Q1 - IQR * 1.5
t = Q3 + IQR * 1.5

Ans3 = train[(train['GPA'] < b) | (train['GPA'] > t)]['GPA'].mean()

# print(round(Ans3, 2)) # 답: 2.33