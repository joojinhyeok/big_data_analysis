import pandas as pd
import numpy as np

# 데이터 생성 (가상의 이커머스 데이터)
np.random.seed(2023)
n_samples = 2000

data = {
    'CustomerID': np.arange(1000, 1000 + n_samples),
    'JoinDate': pd.date_range(start='2020-01-01', end='2023-12-31', periods=n_samples),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    'Age': np.random.randint(18, 70, n_samples),
    'TotalSpend': np.random.exponential(1000, n_samples) + 100, # 지수분포
    'NumPurchases': np.random.randint(1, 50, n_samples),
    'Email': [f"user{i}@{np.random.choice(['gmail.com', 'naver.com', 'daum.net', 'yahoo.com'])}" for i in range(n_samples)],
    'Satisfaction': np.random.choice([1, 2, 3, 4, 5], n_samples),
    'DiscountUsed': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
}

df = pd.DataFrame(data)

# 타겟(Churn) 생성: 구매 적고, 만족도 낮고, 나이 어리면 이탈 확률 높음
prob = 0.1
prob += np.where(df['TotalSpend'] < 500, 0.3, 0)
prob += np.where(df['Satisfaction'] <= 2, 0.4, 0)
prob += np.where(df['Age'] < 25, 0.1, 0)
prob += np.random.normal(0, 0.1, n_samples)

prob = np.clip(prob, 0, 1)
df['Churn'] = np.random.binomial(1, prob)

# 결측치 심기 (TotalSpend에 50개)
df.loc[np.random.choice(df.index, 50), 'TotalSpend'] = np.nan

# 날짜를 문자열로 변환 (시험환경 흉내)
df['JoinDate'] = df['JoinDate'].astype(str)

# Train / Test 분리
train = df.iloc[:1500]
test = df.iloc[1500:].drop('Churn', axis=1)

train.to_csv('ecommerce_train.csv', index=False)
test.to_csv('ecommerce_test.csv', index=False)

print("준비 완료! ecommerce_train.csv, ecommerce_test.csv")

import pandas as pd

train = pd.read_csv('ecommerce_train.csv')
test = pd.read_csv('ecommerce_test.csv')