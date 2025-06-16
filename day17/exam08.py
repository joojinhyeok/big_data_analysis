# 결측치 수 & 비율 계산
# Titanic 데이터셋에서 Age 컬럼의
# (1) 결측치 개수와
# (2) 전체 대비 결측치 비율(%)을 각각 구하시오
# 단, 비율은 소수 첫째 자리에서 반올림한 정수형으로 출력하시오. 

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')

# print(train.info())

# 결측치 개수
a = train['Age'].isnull().sum()
print(a)

# 결측치 비율
b = round(a / len(train) * 100)

print(b)