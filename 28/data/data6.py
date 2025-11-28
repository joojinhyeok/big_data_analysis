import pandas as pd
import numpy as np

# 데이터 생성 (가상의 취업 예측 데이터)
np.random.seed(2024)
n_samples = 1500

data = {
    'ID': np.arange(1000, 1000 + n_samples),
    'Age': np.random.randint(22, 35, n_samples),
    'Gender': np.random.choice(['M', 'F'], n_samples),
    'GPA': np.random.normal(3.5, 0.4, n_samples), # 학점
    'Major': np.random.choice(['CS', 'Eng', 'Biz', 'Art'], n_samples),
    'Toeic': np.random.normal(800, 100, n_samples),
    'Internship': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]), # 인턴 경험 유무
    'JoinDate': pd.date_range(start='2023-01-01', periods=n_samples)
}

df = pd.DataFrame(data)

# 데이터 다듬기
df['GPA'] = df['GPA'].clip(2.0, 4.5)
df['Toeic'] = df['Toeic'].clip(400, 990).astype(int)

# 타겟 1 (Placement: 취업 여부 0/1) - 분류용
score = (df['GPA'] * 100) + (df['Toeic'] * 0.5) + (df['Internship'] * 50) + np.random.normal(0, 50, n_samples)
df['Placement'] = np.where(score > 750, 1, 0)

# 타겟 2 (Salary: 연봉) - 회귀용
df['Salary'] = (3000 + df['Placement'] * 1000 + df['Toeic'] * 2 + np.random.normal(0, 200, n_samples)).astype(int)

# 결측치 심기 (GPA에 50개)
df.loc[np.random.choice(df.index, 50), 'GPA'] = np.nan

# 날짜를 문자열로 변환 (시험환경 흉내)
df['JoinDate'] = df['JoinDate'].astype(str)

# Train / Test 분리
train = df.iloc[:1200]
test = df.iloc[1200:].drop(['Placement', 'Salary'], axis=1) # 타겟 2개 다 드랍

# 파일 저장
train.to_csv('job_train.csv', index=False)
test.to_csv('job_test.csv', index=False)

print("준비 완료! job_train.csv, job_test.csv")