# 6. 조건부 평균 계산
# Titanic 데이터셋에서 "Fare"가 30 이상인 승객들에 대해
# Age의 평균을 계산하시오
# 결과는 소수 첫째자리에서 반올림하여 정수형으로 출력하시오.

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')

# Fare가 30이상인 조건을 condition에 선언
condition = train['Fare'] >= 30

# round로 첫째자리에서 반올림
mean_age = round(train[condition]['Age'].mean())

print(int(mean_age))