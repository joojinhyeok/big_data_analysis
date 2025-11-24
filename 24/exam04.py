# train.csv에서 Name 컬럼을 보면 Mr., Mrs., Miss. 같은 호칭이 있어. Name 컬럼에서 **'Mr'**와 **'Mrs'**가 포함된 승객의 수를 각각 구하고,
# 'Mr'가 포함된 승객 수와 'Mrs'가 포함된 승객 수의 합을 구하시오.

import pandas as pd

train = pd.read_csv('csv/train.csv')

# print(train.info())

# str.contains()로 True/False로 된 리스트가 생성
# Mr.와 Mrs.를 각각 구하고 싶으면 \.을 붙여서 풀이
# mr = train['Name'].str.contains('Mr\.')
# mrs = train['Name'].str.contains('Mrs\.')
mr = train['Name'].str.contains('Mr')
mrs = train['Name'].str.contains('Mrs')

Ans = mr.sum() + mrs.sum()

print(Ans) # 답: 776