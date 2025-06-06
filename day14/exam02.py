# *** 1유형 ***
# 문제
# 1. 나이(Age)의 결측치를 Pclass별 평균 나이로 채우시오
# 2. 결과는 Sex, Pclass별 평균 나이를 구하시오
# 3. 결과는 Sex, Pclass, Average_Age 컬럼으로 구성하시오
# 4. 평균 나이가 낮은 순으로 정렬하시오

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')

print(train.info())

# 1. 나이(Age)의 결측치를 Pclass별 평균 나이로 채우시오
train['Age'] = train['Age'].fillna(train.groupby('Pclass')['Age'].transform('mean'))

# print(train['Pclass'].mean())
# print(train['Age']) -> 확인용

# 2. 결과는 Sex, Pclass별 평균 나이를 구하시오
df = train.groupby(['Sex', 'Pclass'])['Age'].mean().reset_index()

# 3. 결과는 Sex, Pclass, Average_Age 컬럼으로 구성하시오
df = df.rename(columns={'Age' : 'Average_Age'})

# 4. 평균 나이가 낮은 순으로 정렬하시오
df = df.sort_values(by="Average_Age", ascending=True)

print(df)