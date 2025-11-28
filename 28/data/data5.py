import pandas as pd
import numpy as np

# 데이터 생성 (가상의 온라인 강의 수강 데이터)
np.random.seed(2024)
n_samples = 1000

data = {
    'UserID': np.arange(1000, 1000 + n_samples),
    'Course': np.random.choice(['Python', 'DataScience', 'AI', 'WebDev'], n_samples),
    'JoinDate': pd.date_range(start='2023-01-01', periods=n_samples),
    'StudyHours': np.random.normal(50, 15, n_samples), # 공부 시간
    'Score': np.random.randint(40, 100, n_samples),    # 시험 점수
    'Tags': np.random.choice(['Easy,Fun', 'Hard,Deep', 'Fun,Quick', 'Hard,Theory'], n_samples), # 리뷰 태그
    'Device': np.random.choice(['PC', 'Mobile'], n_samples)
}

df = pd.DataFrame(data)

# 결측치 심기 (StudyHours에 50개)
df.loc[np.random.choice(df.index, 50), 'StudyHours'] = np.nan

# 파일 저장
df.to_csv('edu_train.csv', index=False)

print("준비 완료! edu_train.csv")