import pandas as pd

"""
📌 전처리 3단계 실습

1단계: 결측치 처리
- 결측치는 데이터 누락으로, 분석 정확도에 영향을 줌
"""

# 데이터 불러오기 -> 그냥 테이블이라고 생각
df = pd.read_csv('C:/csv/test.csv')

# 1. Age 컬럼의 결측치를 평균값으로 대체
df['Age'] = df['Age'].fillna(df['Age'].mean())
print("Age", df['Age'])
# 평균값(30.27...)으로 결측치가 채워짐

# 2. Fare 컬럼의 결측치를 중앙값으로 대체
df['Fare'] = df['Fare'].fillna(df['Fare'].median()) # .fillna() 결측치 채우는 함수
print("Fare", df['Fare'])
# 결측치 1개가 중앙값으로 대체되어 NaN 없음

# 3. Cabin 컬럼의 결측치를 'Unknown' 문자열로 대체
df['Cabin'] = df['Cabin'].fillna('Unknown')
print("Cabin", df['Cabin'])
# 대부분 결측치였던 Cabin 컬럼이 'Unknown'으로 채워짐

"""
✅ 1단계 완료 (결측치 처리)

------------------------------------------------------

2단계: 이상치 처리 (Fare 기준, IQR 방식)

- 이상치는 데이터 분포에서 극단적으로 큰 값이나 작은 값
- 분석 결과에 큰 영향을 줄 수 있기 때문에 제거 또는 변환이 필요
"""

# 1. IQR(Interquartile Range, 사분위 범위) 계산
Q1 = df['Fare'].quantile(0.25)  # 1사분위수
Q3 = df['Fare'].quantile(0.75)  # 3사분위수
IQR = Q3 - Q1                   # IQR = Q3 - Q1(= 중앙 50% 데이터가 퍼져 있는 범위)
print("**IQR** = ", IQR)

# 2. 이상치 기준 설정
lower = Q1 - 1.5 * IQR  # 하한선(이 값보다 작으면 이상치)
upper = Q3 + 1.5 * IQR  # 상한선(이 값보다 크면 이상치)

# 3. 이상치만 추출하여 개수 확인
outliers = df[(df['Fare'] < lower) | (df['Fare'] > upper)]
print("이상치 개수:", len(outliers))
print(outliers[['Fare']].head())  # 이상치 상위 5개 출력

# 4. 이상치 제거 (Fare 컬럼 기준)
df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]
# 이상치 55개가 제거된 새로운 df로 갱신됨

"""
✅ 2단계 완료 (이상치 처리)

------------------------------------------------------

3단계: 인코딩 (Encoding)

- 머신러닝 모델은 문자열을 처리할 수 없으므로
  범주형 변수(Categorical)를 숫자형으로 변환해야 함
- 여기서는 One-Hot Encoding 사용
"""

# One-Hot Encoding (drop_first=True: 기준값 제거로 다중공선성 방지)
# get_dumies()는 "범주형(카테고리형) 데이터를 숫자로 변환"해주는 함수
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

print("3단계 인코딩", df.head())
# 결과 컬럼 예시: Sex_male, Embarked_Q, Embarked_S
