# 3. 정렬 후 추출
# Titanic 데이터셋에서 "Fare(운임)"가 높은 순으로 정렬한 뒤,
# Fare가 가장 높은 5명의 Name과 Fare을 출력하시오.
# 단, Fare 내림차순으로 정렬된 상태여야 하며, 결과는 DataFrame 형태로 출력하시오.

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')

# Fare를 기준으로 내림차순 정렬하여 새로운 DataFrame 생성
sort_train = train.sort_values(by='Fare', ascending=False)

# 정렬된 DataFrame에서 상위 5명만 추출 (Name, Fare 컬럼만 선택)
top5 = sort_train[['Name', 'Fare']].iloc[:5]

print(top5)
