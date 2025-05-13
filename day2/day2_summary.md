## 📅 Day 2 – 전처리 실습 요약

### ✅ 학습 목표

* 결측치(Missing Value) 처리
* 이상치(Outlier) 탐지 및 제거 (IQR 방식)
* 범주형 변수 인코딩 (One-Hot Encoding)
* 실전 기출 스타일 실습 1제 풀이

---

### 📁 사용한 데이터

* `test.csv` (Kaggle Titanic Competition 테스트 데이터)

---

### 🔍 전처리 단계 요약

| 단계  | 내용     | 적용 컬럼            | 처리 방식                                |
| --- | ------ | ---------------- | ------------------------------------ |
| 1단계 | 결측치 처리 | Age, Fare, Cabin | 평균, 중앙값, 문자열 'Unknown'               |
| 2단계 | 이상치 처리 | Fare             | IQR 방식으로 제거                          |
| 3단계 | 인코딩    | Sex, Embarked    | One-Hot Encoding (`drop_first=True`) |

---

### 🧪 실전 스타일 실습 문제

#### 🎯 문제

**다음 전처리 과정을 수행하시오 (Titanic 데이터 기준)**

1. `Age` 컬럼의 결측치는 평균으로, `Fare`는 중앙값으로 채움
2. `Cabin` 결측치는 문자열 `"Unknown"`으로 대체
3. `Fare`의 이상치를 IQR 방식으로 제거
4. `Sex`, `Embarked` 컬럼을 One-Hot Encoding 수행
5. 최종 컬럼 수를 출력하시오

#### 💻 정답 코드 예시

```python
import pandas as pd

# 데이터 로드
df = pd.read_csv("test.csv")

# 1. 결측치 처리
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Cabin'] = df['Cabin'].fillna('Unknown')

# 2. 이상치 제거 (Fare 기준 IQR)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]

# 3. One-Hot Encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 최종 컬럼 수
print("컬럼 수:", df.shape[1])
```

---

### 🧠 오늘의 개념 한 줄 요약

* `fillna()`는 결측치를 채우는 함수 (평균, 중앙값 등)
* IQR 이상치 탐지는 `Q3 + 1.5*IQR`, `Q1 - 1.5*IQR` 기준
* `get_dummies()`는 문자 → 숫자 인코딩하는 함수
* `drop_first=True`는 기준값 1개 제거하여 다중공선성 방지

---

원하면 Day 3부터는 \*\*스케일링, 파생변수, 데이터 분리(train/test)\*\*로 넘어갈 수 있어!
필요하면 `"day_3도 준비해줘"`라고 말해줘 😊
