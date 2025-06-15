# 1유형 문제
# Titanic 데이터셋
# --------------------------------------------------------------------------------------
# 1. Age 컬럼에 결측치가 존재한다.
#    Pclass별 평균 나이로 Age의 결측치를 채우시오.
# --------------------------------------------------------------------------------------

# 데이터 불러오기
import pandas as pd

train = pd.read_csv('C:/csv/train.csv')

# Pclass별 평균 나이로 Age 결측치 채우기

# print(train.info())

train['Age'] = train['Age'].fillna(train.groupby('Pclass')['Age'].transform('mean'))

# print(train['Age'].isnull().sum())

# --------------------------------------------------------------------------------------
# 2. Titanic 데이터셋의 Fare 컬럼에 이상치가 존재한다.
#    1사분위수(Q1)보다 1.5 IQR만큼 아래이거나,
#    3사분위수(Q3)보다 1.5 IQR만큼 초과하는 값은 이상치로 간주한다.
#    이상치를 ** Fare 컬럼의 중앙값(median) **으로 대체하시오.
# --------------------------------------------------------------------------------------
Q1 = train['Fare'].quantile(0.25)   # 1사분위수 (25%)
Q3 = train['Fare'].quantile(0.75)   # 3사분위수 (75%)

IQR = Q3 - Q1   # IQR 계산

lower_bound = Q1 - 1.5 * IQR  # 이상치 하한
upper_bound = Q3 + 1.5 * IQR  # 이상치 상한

median_fare = train['Fare'].median()   # Fare의 중앙값

# train.loc[조건, 'Fare'] = median_fare
# -> 조건을 만족하는 행들의 'Fare' 값만 지정해서 median_fare값으로 채움 
train.loc[(train['Fare'] < lower_bound) | (train['Fare'] > upper_bound), 'Fare'] = median_fare

# print(train['Fare'].shape)

# --------------------------------------------------------------------------------------
# 3. Titanic 데이터셋의 Embarked 컬럼은 승선한 항구 정보를 의미하며,
#    일부 결측치가 존재한다.
#    가장 많이 나타나는 값(최빈값, mode)으로 결측치를 채우시오.
# --------------------------------------------------------------------------------------
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

print(train['Embarked'])

print(train['Embarked'].isnull().sum())