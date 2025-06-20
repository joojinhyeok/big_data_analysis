# 실전 모의고사


import pandas as pd
import numpy as np
from scipy import stats

train = pd.read_csv('C:/csv/train.csv')
test = pd.read_csv('C:/csv/test.csv')

# 3유형
# Titanic 데이터셋에서 남성과 여성 승객의 요금(Fare) 분포가
# 유의미하게 다른지를 검정하려고 한다. 아래 문항에 답하시오.

# print(train.info())

# Q1
# 남성과 여성의 평균 Fare를 각각 출력하시오.
# (단, Sex 컬럼이 문자열이면 LabelEncoder 등으로 처리해도 무방)

group1 = train[train['Sex']=='male']['Fare'].mean()
group2 = train[train['Sex']=='female']['Fare'].mean()

print('남성 평균 Fare: ', group1)
print('여성 평균 Fare: ', group2)

# Q2
# 남성과 여성의 Fare 분산이 같은지 여부를 F-검정을 통해 판단하시오.
# F-통계량과 p-value를 출력하시오.
# (분산 큰 값 / 작은 값 순서로 나누고 계산)


# Q3
# 분산이 동일하다고 가정한 독립표본 t-검정을 수행하시오.
# 검정통계량과 p-value를 출력하시오.
# (유의수준 0.05, 결과 해석은 생략)