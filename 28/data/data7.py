import pandas as pd
import numpy as np

# 데이터 생성 (가상의 전자제품 쇼핑 데이터)
np.random.seed(2025)
n_samples = 1200

data = {
    'OrderID': np.arange(1000, 1000 + n_samples),
    'Date': pd.date_range(start='2023-01-01', periods=n_samples, freq='H'), # 시간 단위
    'Category': np.random.choice(['Laptop', 'Mobile', 'Tablet', 'Watch'], n_samples),
    'Price': np.random.randint(50, 200, n_samples) * 10000, # 가격
    'Qty': np.random.randint(1, 6, n_samples), # 수량
    'MemberType': np.random.choice(['VIP', 'Gold', 'Silver', 'Bronze'], n_samples),
    'Gender': np.random.choice(['M', 'F'], n_samples),
    'Age': np.random.randint(20, 60, n_samples),
    'ReviewScore': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.1, 0.2, 0.3, 0.3])
}

df = pd.DataFrame(data)

# 결측치 심기 (Price에 30개)
df.loc[np.random.choice(df.index, 30), 'Price'] = np.nan

# 파일 저장
df.to_csv('shop_train.csv', index=False)

print("준비 완료! shop_train.csv")