# 📅 Day 14 – 1유형 실전 문제 집중 훈련

## ✅ 학습 목표

- 1유형 실전 문제를 통해 pandas 핵심 문법 숙련
- Titanic 데이터셋을 활용한 그룹별 집계, 결측치 처리, 정렬 등 실습
- groupby(), reset_index(), fillna(), sort_values() 등 실전 문법 복습

---

## 🧪 실전 문제 1

### 📌 문제
1. Sex와 Pclass별 생존률(Survived 평균)을 구하시오  
2. 결과는 Sex, Pclass, Survival_Rate 컬럼으로 구성하시오  
3. 생존률이 높은 순으로 정렬하시오

### 💻 코드
```python
import pandas as pd

train = pd.read_csv('C:/csv/train.csv')
print(train.info())

# 1. Sex와 Pclass별 생존률 계산
df = train.groupby(['Sex', 'Pclass'])['Survived'].mean().reset_index()

# 2. 컬럼 이름 변경
df = df.rename(columns={'Survived': 'Survival_Rate'})

# 3. 생존률 높은 순으로 정렬
df = df.sort_values(by='Survival_Rate', ascending=False)

print(df)
````

---

## 🧪 실전 문제 2

### 📌 문제

1. Age의 결측치를 Pclass별 평균 나이로 채우시오
2. Sex, Pclass별 평균 나이를 구하시오
3. 결과는 Sex, Pclass, Average\_Age 컬럼으로 구성하시오
4. 평균 나이가 낮은 순으로 정렬하시오

### 💻 코드

```python
import pandas as pd

train = pd.read_csv('C:/csv/train.csv')
print(train.info())

# 1. Pclass별 평균 나이로 Age 결측치 채우기
train['Age'] = train['Age'].fillna(train.groupby('Pclass')['Age'].transform('mean'))

# 2. Sex, Pclass별 평균 나이 계산
df = train.groupby(['Sex', 'Pclass'])['Age'].mean().reset_index()

# 3. 컬럼 이름 변경
df = df.rename(columns={'Age': 'Average_Age'})

# 4. 평균 나이가 낮은 순으로 정렬
df = df.sort_values(by='Average_Age', ascending=True)

print(df)
```

---

## 🧠 복습한 주요 문법 요약

| 함수                  | 설명                                    |
| ------------------- | ------------------------------------- |
| `groupby()`         | 특정 컬럼 기준으로 그룹화 후 집계                   |
| `reset_index()`     | groupby 결과를 일반 데이터프레임으로 복원            |
| `fillna()`          | 결측치를 원하는 값으로 대체                       |
| `transform('mean')` | 그룹 기준으로 각 행에 평균값 대응                   |
| `sort_values()`     | 원하는 컬럼 기준으로 정렬 (ascending=True: 오름차순) |
| `rename()`          | 컬럼명 변경                                |

---

## ✅ 실습 결과 정리

* 1유형 문제에 자주 나오는 핵심 스킬 집중 훈련 완료
* Titanic 데이터 기반의 생존률/나이 분석 실전 대비 연습