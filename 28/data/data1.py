import pandas as pd
import numpy as np

# 데이터 생성 (가상의 중고차 데이터)
np.random.seed(2023)
n_samples = 1200

data = {
    'id': np.arange(1000, 1000 + n_samples),
    'brand': np.random.choice(['Hyundai', 'Kia', 'BMW', 'Audi'], n_samples),
    'year': np.random.randint(2010, 2023, n_samples),
    'transmission': np.random.choice(['Automatic', 'Manual'], n_samples, p=[0.7, 0.3]),
    'mileage': np.random.normal(50000, 20000, n_samples),
    'fuelType': np.random.choice(['Petrol', 'Diesel', 'Hybrid'], n_samples),
    'tax': np.random.randint(0, 500, n_samples),
    'mpg': np.random.normal(40, 10, n_samples),
    'engineSize': np.random.choice([1.0, 1.6, 2.0, 3.0], n_samples)
}

df = pd.DataFrame(data)

# 타겟(price) 생성: 연식 최신, 주행거리 짧음, 엔진 큼 -> 가격 높음
df['price'] = (
    20000 
    + (df['year'] - 2010) * 1000 
    - (df['mileage'] * 0.1) 
    + (df['engineSize'] * 5000) 
    + np.where(df['brand'] == 'BMW', 5000, 0)
    + np.random.normal(0, 2000, n_samples)
)
df['price'] = df['price'].clip(lower=1000).astype(int) # 음수 방지

# 결측치 심기 (mpg에 결측치 생성)
df.loc[np.random.choice(df.index, 30), 'mpg'] = np.nan

# Train / Test 분리 (900개 학습, 300개 평가)
train = df.iloc[:900]
test = df.iloc[900:].drop('price', axis=1)

# 파일 저장
train.to_csv('car_train.csv', index=False)
test.to_csv('car_test.csv', index=False)

print("준비 완료! car_train.csv, car_test.csv 생성됨")