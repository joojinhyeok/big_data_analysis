# [문제 1] 1유형 - 문자열 필터링 & 그룹 비교
# Name 컬럼에서 호칭이 'Mr.'(점 포함) 인 그룹과 'Miss.'(점 포함) 인 그룹을 각각 추출하시오. 
# 두 그룹의 **Age 평균의 차이(절대값)**를 구하시오. (단, 결측치는 제거하지 않고 계산하며, 정답은 소수점 넷째 자리에서 반올림하여 셋째 자리까지 출력)

import pandas as pd
train = pd.read_csv('csv/train.csv')

mr = train[train['Name'].str.contains('Mr\.')]
miss = train[train['Name'].str.contains('Miss\.')]

# print(mr, mrs)

Ans = abs(mr['Age'].mean() - miss['Age'].mean())

print(round(Ans, 3)) # 답: 10.594