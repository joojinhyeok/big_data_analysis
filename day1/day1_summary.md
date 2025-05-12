---

## 📅 Day 1 – Titanic EDA 기본 실습 요약

### ✅ 학습 목표

* `pandas` 기본 사용법 익히기
* `.read_csv()`, `.shape`, `.info()`, `.describe()` 등 데이터 구조 파악
* `isnull()`, `groupby()`, 조건 필터링 연습
* 실전 문제 3제 풀이

---

### 📁 사용한 데이터

* `test.csv` (Kaggle Titanic Competition 테스트 데이터)
* `train.csv` (분석용 데이터 – 생존 여부 포함)

---

### 🔍 주요 함수 요약

| 함수명                 | 설명                     |
| ------------------- | ---------------------- |
| `pd.read_csv()`     | CSV 파일 불러오기            |
| `df.shape`          | (행, 열) 형태의 구조 확인       |
| `df.info()`         | 컬럼별 데이터 타입 및 결측치 정보 확인 |
| `df.describe()`     | 수치형 데이터 요약 통계 확인       |
| `df.isnull().sum()` | 컬럼별 결측치 개수 확인          |
| `df.groupby()`      | 그룹별 통계 분석              |
| `df[...]` 조건 필터     | 조건에 맞는 행 추출            |

---

### 🧪 실전 문제 풀이

#### 🎯 문제 1

**Fare가 100 이상인 승객 수는 몇 명인가?**

```python
df[df['Fare'] >= 100].shape[0]
```

---

#### 🎯 문제 2

**3등급(Pclass == 3) 중에서 여성(Sex == 'female')의 생존율은?**

```python
df[(df['Pclass'] == 3) & (df['Sex'] == 'female')]['Survived'].mean()
```

---

#### 🎯 문제 3

**가장 요금을 많이 낸 승객의 이름은?**

```python
df[df['Fare'] == df['Fare'].max()]['Name'].values[0]
```

---

### 📘 오늘의 개념 한 줄 요약

* `df = pd.read_csv()` → csv 파일을 불러와 pandas DataFrame으로 저장
* `df`는 곧 “데이터 테이블 그 자체”로 생각하면 됨
* `isnull()`은 결측치 파악, `describe()`는 수치 요약
* 실전 문제는 groupby, 필터링, max 등 자주 사용

---
