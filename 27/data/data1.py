import pandas as pd
import numpy as np

# 데이터 생성 (가상의 직원 퇴사 데이터)
np.random.seed(0)
n_samples = 1500

data = {
    'EmployeeID': np.arange(1000, 1000 + n_samples),
    'Age': np.random.randint(20, 60, n_samples),
    'Department': np.random.choice(['Sales', 'R&D', 'HR'], n_samples, p=[0.4, 0.5, 0.1]),
    'DistanceFromHome': np.random.randint(1, 30, n_samples),
    'Education': np.random.choice([1, 2, 3, 4], n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'JobLevel': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.4, 0.3, 0.2, 0.08, 0.02]),
    'MonthlyIncome': np.random.normal(5000, 2000, n_samples),
    'OverTime': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
    'YearsAtCompany': np.random.randint(0, 20, n_samples)
}

df = pd.DataFrame(data)

# 타겟(Attrition) 생성: 야근(OverTime) 많고, 월급 적고, 근속연수 짧으면 퇴사 확률 높음
prob = 0.1
prob += np.where(df['OverTime'] == 'Yes', 0.4, 0)
prob += np.where(df['MonthlyIncome'] < 3000, 0.2, 0)
prob += np.where(df['YearsAtCompany'] < 3, 0.1, 0)
prob += np.random.normal(0, 0.1, n_samples) # 노이즈 추가

# 확률을 0~1 사이로 맞춤
prob = np.clip(prob, 0, 1)
df['Attrition'] = np.random.binomial(1, prob)

# 결측치 심기 (MonthlyIncome에 일부러 NaN 생성)
df.loc[np.random.choice(df.index, 20), 'MonthlyIncome'] = np.nan

# Train / Test 분리 (1200개 학습, 300개 평가)
train = df.iloc[:1200]
test = df.iloc[1200:].drop('Attrition', axis=1)

# 파일 저장
train.to_csv('employee_train.csv', index=False)
test.to_csv('employee_test.csv', index=False)

print("파일 생성 완료! employee_train.csv, employee_test.csv")