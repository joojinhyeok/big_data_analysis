# 1. 그룹별 통계
# Titanic 데이터셋에서 Sex와 Pclass 별로 생존률(Survived)의 평균을 구하시오
# 단, 컬럼 이름은 SurvivalRate로 지정하고, 평균 생존률이 낮은 순서대로 정렬하시오

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')

print(train.info())

# reset_index()
# groupby()의 결과가 Series나 Index가 여러 개인 형태일 때,
# 그걸 일반적인 DataFrame으로 바꿔주는 함수
a = train.groupby(['Sex', 'Pclass'])['Survived'].mean().reset_index()

a = a.rename(columns = {'Survived' : 'SurvivalRate'})

a = a.sort_values(by='SurvivalRate', ascending=True)

print(a)