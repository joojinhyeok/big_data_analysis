import pandas as pd

df = pd.read_csv('C:/csv/tested.csv')
print(df.shape) # df.shape -> 데이터 크기 출력
print(df.info())    # df.info() -> 컬럼별 데이터 요약
print(df.describe())    # df.describe() -> 수치형 컬럼 요약 통계
print(df.isnull().sum())    # df.isnull().sum() -> 컬럼별 결측치 수

print("-----------", df.sample())   # 무작위로 샘플(일부 행) 추출

