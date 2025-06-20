# 실전 모의고사

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')
test = pd.read_csv('C:/csv/test.csv')

# 1유형
# print(train.info())
# Q1. Embarked 컬럼의 결측치를 가장 많이 나타나는 값으로 채우고,
#     채운 후 s의 개수를 출력하시오.
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
result = (train['Embarked'] == 'S').sum()

# print(result) # 답: 646

# Q2. Pclass별 Age의 평균을 구하고, 평균 Age가 가장 높은 Pclass를 출력하시오.
a = train.groupby('Pclass')['Age'].mean().sort_values()
# print(a) --> 답: 1

# Q3. Fare 컬럼의 이상치를 IQR 방식으로 탐지하고, 이상치의 개수를 출력하시오.
#     (IQR 방식: Q1 - 1.5IQR, Q3 + 1.5IQR)
Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - (1.5 * IQR)
upper = Q3 + (1.5 * IQR)

result = train[(train['Fare'] < lower) | (train['Fare'] > upper)]

print(len(result)) # 답: 116