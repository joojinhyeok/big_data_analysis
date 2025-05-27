## 📅 Day 3 – 피벗, 병합, 전처리 실습 요약

### ✅ 학습 목표

* `pivot_table`과 `crosstab`을 이용한 다차원 집계
* `merge`, `concat`을 통한 데이터프레임 병합
* 타입 변환 및 이상치 제거 실습
* 실전 기출 스타일 실습 1제 풀이

---

### 📁 사용한 데이터

* 자체 제작 샘플 데이터 (사용자 정보, 주문 정보 등)

---

### 🔍 주요 함수 요약

| 함수명                | 설명                                                     |
| ------------------ | ------------------------------------------------------ |
| `pd.pivot_table()` | 그룹별로 수치 데이터를 집계하여 테이블 형태로 재구성                          |
| `pd.crosstab()`    | 범주형 변수 간의 교차 빈도표 생성                                    |
| `pd.merge()`       | SQL JOIN과 유사한 방식으로 두 DF 병합 (inner, left, right, outer) |
| `pd.concat()`      | 행 또는 열 방향으로 DF 결합 (`axis=0/1`)                         |
| `astype()`         | 컬럼 데이터 타입 변환                                           |
| 조건 필터링             | 이상치 제거, 범위 지정 등 데이터 선택                                 |

---

### 🧪 실전 스타일 실습 문제

#### 🎯 문제

**다음 조건에 맞게 데이터 전처리 및 피벗 테이블을 작성하시오.**

1. `orders`와 `users` 데이터를 `user_id` 기준으로 병합
2. `gender`를 행, `product`를 열로 하여 구매 `amount`의 합계를 피벗 테이블로 출력
3. 제품을 구매하지 않은 항목은 `0`으로 표시

#### 💻 정답 코드 예시

```python
import pandas as pd

# 주문 데이터
orders = pd.DataFrame({
    'user_id': [1, 2, 1, 3],
    'product': ['A', 'B', 'A', 'C'],
    'amount': [2, 1, 1, 5]
})

# 사용자 데이터
users = pd.DataFrame({
    'user_id': [1, 2, 3],
    'gender': ['F', 'M', 'F']
})

# 병합
merged = pd.merge(orders, users, on='user_id')

# 피벗 테이블 생성
pivot = pd.pivot_table(
    merged,
    values='amount',
    index='gender',
    columns='product',
    aggfunc='sum',
    fill_value=0
)

print(pivot)
```

---

### 🧠 오늘의 개념 한 줄 요약

* `pivot_table()`은 다차원 집계를 깔끔하게 정리할 수 있다.
* `merge()`는 공통 컬럼을 기준으로 DF 병합할 때 사용한다.
* `concat()`은 단순히 행/열 방향으로 DF를 이어 붙인다.
* `astype()`으로 타입을 정확히 맞춰야 분석이 가능하다.
* 이상치 제거는 조건 필터링으로 간단히 구현할 수 있다.