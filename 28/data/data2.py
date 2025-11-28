import pandas as pd
import numpy as np

# 데이터 생성 (가상의 의료비 데이터)
np.random.seed(2023)
n_samples = 1300

data = {
    'id': np.arange(1000, 1000 + n_samples),
    'age': np.random.randint(18, 65, n_samples),
    'sex': np.random.choice(['male', 'female'], n_samples),
    'bmi': np.random.normal(30, 5, n_samples),
    'children': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.4, 0.25, 0.2, 0.1, 0.03, 0.02]),
    'smoker': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
    'region': np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples),
}

df = pd.DataFrame(data)

# 타겟(charges) 생성: 흡연자, 고령, 비만일수록 비쌈
base_charge = 3000 + (df['age'] * 200)
bmi_charge = (df['bmi'] - 25) * 300
smoke_charge = np.where(df['smoker'] == 'yes', 20000, 0)
noise = np.random.normal(0, 1000, n_samples)

df['charges'] = base_charge + bmi_charge + smoke_charge + noise
df['charges'] = df['charges'].abs() # 음수 방지

# 결측치 심기 (bmi에 50개)
df.loc[np.random.choice(df.index, 50), 'bmi'] = np.nan

# Train / Test 분리 (1000개 학습, 300개 평가)
train = df.iloc[:1000]
test = df.iloc[1000:].drop('charges', axis=1)

# 파일 저장
train.to_csv('medical_train.csv', index=False)
test.to_csv('medical_test.csv', index=False)

print("준비 완료! medical_train.csv, medical_test.csv")