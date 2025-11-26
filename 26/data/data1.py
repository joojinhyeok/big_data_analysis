import pandas as pd
import numpy as np

# 데이터 생성 (가상의 자전거 대여 데이터)
np.random.seed(42)
n_samples = 1000

data = {
    'datetime': pd.date_range(start='2022-01-01', periods=n_samples, freq='H'),
    'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
    'holiday': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'workingday': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'weather': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Snow'], n_samples, p=[0.6, 0.25, 0.1, 0.05]),
    'temp': np.random.normal(20, 5, n_samples),
    'humidity': np.random.uniform(30, 90, n_samples),
    'windspeed': np.random.exponential(10, n_samples)
}

df = pd.DataFrame(data)

# 타겟(count) 생성: 온도, 계절 등에 영향받게 설정 (회귀 문제용)
df['count'] = (
    100 
    + 5 * df['temp'] 
    - 0.5 * df['humidity'] 
    + np.where(df['season'] == 'Fall', 50, 0) 
    + np.where(df['weather'] == 'Rain', -30, 0)
    + np.random.normal(0, 20, n_samples) # 노이즈
).astype(int)

df['count'] = df['count'].clip(lower=0) # 음수 제거

# 결측치 일부러 생성 (전처리 연습용)
df.loc[np.random.choice(df.index, 50), 'humidity'] = np.nan

# Train / Test 분리
train = df.iloc[:800]
test = df.iloc[800:].drop('count', axis=1) # 타겟 제거
test_id = df.iloc[800:]['datetime'] # 제출용 ID (날짜)

# 파일 저장
train.to_csv('bike_train.csv', index=False)
test.to_csv('bike_test.csv', index=False)

print("파일 생성 완료! bike_train.csv, bike_test.csv 확인해봐!")
