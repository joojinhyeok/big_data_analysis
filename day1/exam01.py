import pandas as pd

df = pd.read_csv('C:/csv/test.csv') # csv파일 불러옴

# 문제 1
# Fare가 100 이상인 승객 수는 몇 명인가?
print(df[df['Fare'] >= 100].shape[0])

# df[df['Fare] > = 100] -> Fare가 100 이상인 행만 필터링
# .shape[0] -> 그 필터링된 행의 개수(= 승객 수)

# .shape는??
# df.shape -> (행, 열)
# df.shape[0] -> 행 개수(레코드 수)
# df.shape[1] -> 열 개수