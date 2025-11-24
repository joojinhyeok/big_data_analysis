# Embarked(탑승 항구) 별로 **평균 운임(Fare)**을 구했을 때, 가장 평균 운임이 높은 항구를 찾으시오.
# 그 **'가장 비싼 항구'**에서 탑승한 승객들 중, **Sex(성별)이 'male'(남성)**인 사람들의 **Age(나이) 중앙값(median)**을 구하시오.
# 정답은 **소수점 버리고 정수(int)**로 출력하시오.

import pandas as pd

train = pd.read_csv('csv/train.csv')

t = train.groupby('Embarked')['Fare'].mean()

# print(t) -> C 항구가 가장 높음

Ans = train[(train['Embarked'] == 'C') & (train['Sex'] == 'male')]['Age'].median()

print(int(Ans)) # 답: 30