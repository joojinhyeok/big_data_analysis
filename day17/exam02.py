# 2. 조건 필터링
# Titanic 데이터셋에서 Pclass가 1등급이며, Sex가 female인 승객들 중
# "평균 생존률(Survived의 평균)"을 구하시오.
# 단, 결과는 정수형으로 변환하여 출력하시오.

import pandas as pd

# 데이터 불러오기
train = pd.read_csv('C:/csv/train.csv')

# 조건: 1등급(Pclass==1) 이고 성별이 여성(Sex=='female')인 경우
# 조건을 condition 변수에 선언
condition = (train['Pclass'] == 1) & (train['Sex'] == 'female')

# 조건을 만족하는 행만 필터링하여 새로운 DataFrame 생성
# filtered라는 새로운 DataFrame이 생성된 것
filtered = train[condition]

# 필터링된 데이터에서 생존률 평균 계산 → 정수형으로 변환
a = int(filtered['Survived'].mean())

# 결과 출력
print(a)

# 한 줄로 작성할 수도 있음
# print(int(train[(train['Pclass'] == 1) & (train['Sex'] == 'female')]['Survived'].mean()))