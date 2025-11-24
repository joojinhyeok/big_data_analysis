# 데이터 불러오기
import pandas as pd

df = pd.read_csv('csv/train.csv')
# df.info()

# 1번 - Age 컬럼의 **결측치(NaN)**를 **전체 데이터의 중앙값(median)**으로 채우시오.
t = df.Age.median()
df['Age'] = df['Age'].fillna(t)
# print("답: ", df.Age.isna().sum())

# 2번 - 보정된 데이터를 기준으로, **성별(Sex)이 'female'**인 승객들의 **평균 나이(Age)**를 구하시오.
result = df[df['Sex'] == 'female']['Age'].mean()
# print("답: ", result)

# 3번 - 정답은 소수점 셋째 자리에서 반올림하여 소수점 둘째 자리까지 출력하시오. (예: 28.123 -> 28.12)
print("답: ", round(result, 2))
