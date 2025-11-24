# gemini 문제

# train.csv 데이터에서 **Pclass가 1등급(1)**인 승객들의 Fare(요금) 데이터를 추출하여, 이 데이터의 IQR (3분위수 - 1분위수) 값을 구하시오.
# 정답은 소수점 이하는 모두 버리고 **정수(int)**로 출력하시오.

import pandas as pd

train = pd.read_csv('csv/train.csv')

# print(train.info())

IQR3 = train[train['Pclass'] == 1]['Fare'].quantile(0.75)
IQR1 = train[train['Pclass'] == 1]['Fare'].quantile(0.25)

Ans = int(IQR3 - IQR1)

print(Ans) # 답: 62