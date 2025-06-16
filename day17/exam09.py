# 9. 이상치 탐지 및 처리
# Titanic 데이터셋에서 Fare 컬럼의 이상치를 IQR 방식을 사용해 탐지하고,
# 이상치 값을 Fare 컬럼의 중앙값으로 대체하시오.
# 이상치는 다음 기준을 따른다.
# "1사분위수(Q1) 보다 1.5 IQR만큼 낮거나"
# "3사분위수 (Q3) 보다 1.5 IQR만큼 높은 값"

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')

# 1사분위수, 3사분위수, IQR 계산
Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1

# 이상치 기준값 계산
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# 이상치 조건 (상한보다 크거나 하한보다 작음)
condition = (train['Fare'] > upper) | (train['Fare'] < lower)

# 중앙값 구하기
# a라는 변수에 중앙값 숫자 하나가 저장
a = train['Fare'].median()

# 이상치 위치에 중앙값 대입
# 이상치 값들만 골라서 중앙값으로 대체하는 코드
# loc[] 기본 개념
# df.loc[행 조건, 열 선택]
# -> 특정 행과 열을 선택하거나 수정할 수 있는 인덱서
# ex) df.loc[3] => 3번 인덱스 행 전체 조회
# ex) df.loc[3, 'Fare] => 3번 인덱스 행의 'Fare'값 조회
# ex) df.loc[df['Fare'] > 100, 'Fare'] => Fare > 100인 행의 'Fare'만 가져옴
train.loc[condition, 'Fare'] = a