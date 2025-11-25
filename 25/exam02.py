import pandas as pd
train = pd.read_csv('csv/train.csv')

# [문제 1] 1유형 - 문자열 처리 & 결측치 대체
# Cabin(객실 번호) 컬럼에는 결측치가 많다. Cabin의 결측치(NaN)를 모두 문자 **'N'**으로 채우시오.
# Cabin의 **첫 번째 글자(구역 알파벳)**만 따서 새로운 컬럼 **Cabin_char**를 만드시오. (예: C85 -> C, N -> N)
# **Cabin_char가 'C'**인 승객들의 Fare(요금) 평균을 구하시오. (소수점 둘째 자리 반올림)

# 결측치 N으로 채우기
train['Cabin'] = train['Cabin'].fillna('N')

# .str[]로 첫 번째 글자만 가져오기 
train['Cabin_char'] = train['Cabin'].str[0]

# 'C'인 승객들의 요금 평균 구하기
Ans1 = train[train['Cabin_char'] == 'C']['Fare'].mean()

# print(round(Ans1, 1)) # 답: 100.2

# ========================================================================================================================================

# [문제 2] 1유형 - 파생변수 생성 & 생존율 비교
# 혼자 온 사람과 가족과 온 사람의 생존율을 비교해보자.
# SibSp(형제자매)와 Parch(부모자녀)를 더해서 Family 컬럼을 만드시오.
# Family가 **0명(혼자)**인 그룹과, **1명 이상(가족 동반)**인 그룹의 **생존율(Survived의 평균)**을 각각 구하고,
# 두 그룹 생존율의 **차이(절대값)**를 구하시오. (소수점 셋째 자리 반올림)

train['Family'] = (train['SibSp'] + train['Parch'])

a1 = train[train['Family'] == 0]['Survived'].mean()
a2 = train[train['Family'] >= 1]['Survived'].mean()

Ans2 = abs(a1 - a2)

# print(round(Ans2, 3)) # 답: 0.202
# ========================================================================================================================================

# [문제 3] 3유형 - 독립표본 T-검정 (중요!)
# 생존한 사람(Survived=1)과 사망한 사람(Survived=0)의 **평균 운임(Fare)**이 통계적으로 다른지 확인하고 싶다.
# 두 그룹의 Fare 데이터를 각각 추출하시오.
# **독립표본 T-검정(ttest_ind)**을 수행하여 **검정통계량(t-statistic)**을 구하시오. (단, 두 그룹의 분산은 다르다고 가정한다. equal_var=False) (소수점 둘째 자리 반올림)

# T-검정 먼저 import 하기
from scipy.stats import ttest_ind

# 생존/사망한 사람의 평균 운임 먼저 구하기
s1 = train[train['Survived'] == 1]['Fare']
s0 = train[train['Survived'] == 0]['Fare']

# 두 그룹의 분산이 다르다고 나오면 
# 무조건 equal_var=False 
Ans3 = ttest_ind(s1, s0, equal_var=False)

print(Ans3) # 답: 
