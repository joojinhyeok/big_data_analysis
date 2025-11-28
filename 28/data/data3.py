import pandas as pd
import numpy as np

# 데이터 생성 (가상의 주택 가격 데이터)
np.random.seed(42)
n_samples = 1500

data = {
    'HouseID': np.arange(1000, 1000 + n_samples),
    'YearBuilt': np.random.randint(1950, 2023, n_samples),
    'LotArea': np.random.normal(10000, 5000, n_samples),
    'Neighborhood': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.1, 0.4, 0.3, 0.2]),
    'OverallQual': np.random.choice(range(1, 11), n_samples), # 1~10점
    'Heating': np.random.choice(['GasA', 'GasW', 'Grav', 'Wall'], n_samples),
    'CentralAir': np.random.choice(['Y', 'N'], n_samples, p=[0.9, 0.1]),
    'GrLivArea': np.random.normal(1500, 500, n_samples) # 지상 거주 면적
}

df = pd.DataFrame(data)

# 타겟(SalePrice) 생성
df['SalePrice'] = (
    50000 
    + (df['OverallQual'] * 10000) 
    + (df['GrLivArea'] * 50) 
    + (df['YearBuilt'] * 100) 
    + np.where(df['Neighborhood'] == 'B', 20000, 0)
    + np.random.normal(0, 5000, n_samples)
).astype(int)

# 결측치 심기 (LotArea에 결측치 생성)
df.loc[np.random.choice(df.index, 50), 'LotArea'] = np.nan

# Train / Test 분리 (1200개 학습, 300개 평가)
train = df.iloc[:1200]
test = df.iloc[1200:].drop('SalePrice', axis=1)

# 파일 저장
train.to_csv('housing_train.csv', index=False)
test.to_csv('housing_test.csv', index=False)

print("준비 완료! housing_train.csv, housing_test.csv 생성됨")