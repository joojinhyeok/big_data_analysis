import pandas as pd

df = pd.read_csv('C:/csv/train.csv')  # CSV 파일 불러오기

# 문제 2
# 3등급(Pclass == 3) 중에서 여성(Sex == 'female')의 생존율은?

print(df[(df['Pclass'] == 3) & (df['Sex'] == 'female')]['Survived'].mean())

# df['Pclass'] == 3 → 객실 등급이 3등급인 조건
# df['Sex'] == 'female' → 성별이 여성인 조건
# & 연산자로 두 조건을 모두 만족하는 행만 필터링

# df[(조건1) & (조건2)] → 3등급 + 여성인 승객만 추출
# ['Survived'] → 해당 승객들의 생존 여부만 추출 (0 or 1)
# .mean() → 평균값 계산 → 생존률(=생존자 수 / 총 인원수)
