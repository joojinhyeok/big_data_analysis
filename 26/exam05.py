# [문제 1] 1유형 - 날짜 변환 & 조건 필터링
# train.csv에는 날짜 정보가 없으니 가상의 날짜를 생성하자.
# 2023-01-01부터 시작하는 날짜 컬럼 Date를 만드시오. (승객 순서대로 하루씩 증가)
# Date 컬럼에서 "수요일(Wednesday)" 이면서 "성별이 여성(female)" 인 승객들의 Fare(요금) 평균을 구하시오.
# 정답은 소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력하시오.

import pandas as pd
train = pd.read_csv('csv/train.csv')

# start부터 날짜를 시작해서 periods만큼 만들어줌(여기선 train 만큼 -> 승객 수 만큼)
train['Date'] = pd.date_range(start = '2023-01-01', periods=len(train))

train['Date'] = train['Date'].dt.dayofweek

Ans = train[(train['Date'] == 2) & (train['Sex'] == 'female')]['Fare'].mean()

print(round(Ans, 2)) # 답: 51.3 