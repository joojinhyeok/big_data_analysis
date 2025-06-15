"""
*** 1유형 ***
문제
1. Sex와 Pcalss별 생존률(Survived 평균)을 구하시오
2. 결과는 Sex, Pclass, Survival_Rate 컬럼으로 구성하시오
3. 생존률이 높은 순으로 정렬하시오
"""

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')

# 데이터 구조 및 결측치 확인
print(train.info()) # 데이터프레임 정보 출력(행 수, 컬럼, 데이터 타입, 결측치 확인용)

# ------------------------------------------------------
# 1. Sex와 Pclass별 생존률(Survived 평균)을 구하시오
# ------------------------------------------------------
df = train.groupby(['Sex', 'Pclass'])['Survived'].mean().reset_index()
# - Sex와 Pclass로 그룹화한 후
# - 각 그룹의 생존률(Survived 평균)을 계산
# - reset_index()로 그룹핑 결과를 다시 데이터 프레임 형태로 변환

# ------------------------------------------------------
# (2) 결과는 Sex, Pclass, Survival_Rate 컬럼으로 구성하시오
# ------------------------------------------------------
df = df.rename(columns={'Survived' : 'Survival_Rate'})
# - 계산된 생존률 컬럼 이름을 보기 좋게 'Survival_Rate'로 변경

# ------------------------------------------------------
# 3. 생존률이 높은 순으로 정렬하시오.
# ------------------------------------------------------
df = df.sort_values(by='Survival_Rate', ascending=False)
# - Survival_Rate 기준으로 높은 생존률 순서로 정렬
# - ascending = True면 오름차순 정렬

print(df)