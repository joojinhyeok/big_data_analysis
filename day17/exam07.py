# 7. 다중 조건 필터링 + 그룹 통계
# Titanic 데이터셋에서 "Embarked"가 'S'이고,
# Sex가 'male'인 승객들에 대해
# PClass별 생존률(Survived 평균)을 구하시오
# 컬럼 이름은 SurvivalRate로 지정하고, Pclass 오름차순으로 정렬하시오.

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')

# 조건: Embarked가 'S'이면서, Sex가 'male'인 승객들
condition = (train['Embarked'] == 'S') & (train['Sex'] == 'male')

# 조건에 맞는 데이터만 필터링하여 새로운 DataFrame 생성
filtered = train[condition]

# 필터링된 데이터에서 Pclass별 생존률(Survived 평균) 계산
ps = filtered.groupby('Pclass')['Survived'].mean().reset_index()

# 컬럼명 'Survived' → 'SurvivalRate'로 변경
ps = ps.rename(columns = {'Survived' : 'SurvivalRate'})

# Pclass 기준으로 오름차순 정렬
ps = ps.sort_values(by='Pclass', ascending=True)

print(ps)