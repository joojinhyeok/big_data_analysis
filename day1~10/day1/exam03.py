import pandas as pd

df = pd.read_csv('C:/csv/test.csv')  # CSV 파일 불러오기

# 문제 3
# 가장 요금을 많이 낸 승객의 이름(Name)은 누구인가?

name = df[df['Fare'] == df['Fare'].max()]['Name'].values[0]
print("가장 요금을 많이 낸 사람의 이름:", name)


# df['Fare'].max() → 요금(Fare)의 최대값
# df['Fare'] == df['Fare'].max() → 최대 요금을 낸 승객 필터링
# ['Name'] → 그 승객의 이름만 추출 (Series 형태)
# .values[0] → Series에서 값만 꺼내오기 (인덱스 없이)

