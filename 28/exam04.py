# [1유형] 데이터 핸들링

import pandas as pd

train = pd.read_csv('medical_train.csv')

print(train.info())

# Q1. 조건부 평균 계산
# medical_train.csv 데이터를 사용하시오. bmi 컬럼의 결측치를 평균값으로 대체한 후, smoker가 'yes'이면서 bmi가 30 이상인 사람들의 charges 평균을 구하시오. 
# (정답은 소수점 첫째 자리에서 반올림하여 정수로 출력)

train['bmi'] = train['bmi'].fillna(train['bmi'].mean())

Ans1 = train[(train['smoker'] == 'yes') & (train['bmi'] >= 30)]['charges'].mean()

# print(round(Ans1, 0)) # 답: 33674

# Q2. 순위 및 그룹핑
# region별로 charges가 가장 높은 상위 20% 데이터를 추출하려 한다. 각 지역(region)별로 charges 기준 내림차순 정렬 후, 상위 20%에 해당하는 데이터들의 bmi 평균을 구하시오. 
# 그중 가장 높은 평균 bmi 값을 소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력하시오. 
# (힌트: 데이터 개수가 지역별로 다르니 비율(0.2) 계산 주의)

# 1. 지역별로 charges 기준 '등수' 매기기 (내림차순 = 돈 많은 순 1등)
# method='first'는 동점자 있을 때 그냥 순서대로 등수 매기는 옵션
train['rank'] = train.groupby('region')['charges'].rank(method='first', ascending=False)

# 2. 지역별 '전체 인원수' 구하기
train['total_count'] = train.groupby('region')['charges'].transform('count')

# 3. [핵심] 상위 20% 필터링 (내 등수가 전체 인원의 20%보다 작거나 같으면 합격!)
top_20 = train[train['rank'] <= train['total_count'] * 0.2]

# 4. 이제 남은 애들 가지고 bmi 평균 구하기
result = top_20.groupby('region')['bmi'].mean().max()

print(round(result, 2)) # 답: 31.48 (똑같이 나옴!)