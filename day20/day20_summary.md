# 📘 Day 20 Summary: 3유형 – 통계 검정 (F검정, 합동분산, t검정)

## ✅ 문제 개요
- 데이터셋: `bcc.csv`  
- 목적: 암 환자(2)와 정상인(1)의 **로그 리지스틴 수치** 차이를 통계적으로 검정  
- 방법: F-검정 → 합동 분산 추정량 → 독립표본 t-검정(p-value 계산)

---

## 🧪 (1) F-검정 (분산 차이 검정)

```python
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("data/bcc.csv")
df['log_resistin'] = np.log(df['Resistin'])

group1 = df[df['Classification'] == 1]['log_resistin']
group2 = df[df['Classification'] == 2]['log_resistin']

var1 = group1.var()
var2 = group2.var()

# 자유도 계산
dof1 = len(group1) - 1
dof2 = len(group2) - 1

# 분자의 자유도가 더 크도록 조건 처리
if dof1 > dof2:
    f_stat = var1 / var2
else:
    f_stat = var2 / var1

print('1번 문제:', round(f_stat, 3))  # 예: 1.348
````

> 🔍 **F-검정 해석**:
>
> * F-통계량이 1에 가까우면 → 두 집단의 **분산 차이 없음**
> * **귀무가설(H₀)**: 두 집단의 분산이 같다 → 채택 가능성 높음

---

## 🧪 (2) 합동 분산 추정량 (Pooled Variance)

```python
n1 = len(group1)
n2 = len(group2)

# 합동 분산 공식
pool_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
print('2번 문제:', round(pool_var, 3))  # 예: 0.449
```

> ✅ **공식**:
>
> $$
> $$

Sp^2 = \frac{(n\_1 - 1)s\_1^2 + (n\_2 - 1)s\_2^2}{n\_1 + n\_2 - 2}
]

---

## 🧪 (3) 독립표본 t-검정 + p-value 계산

```python
mean1 = group1.mean()
mean2 = group2.mean()

# T-통계량 수식
t_stat = (mean1 - mean2) / np.sqrt(pool_var * (1/n1 + 1/n2))

# 양측 검정: 절댓값 처리 후 누적 분포 활용
p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n1 + n2 - 2))

print('3번 문제:', round(p_val, 3))  # 예: 0.003
```

> 🔍 **t-검정 해석**:
>
> * p-value < 0.05 → **귀무가설 기각**, 대립가설 채택
> * 즉, **두 집단의 평균은 유의미한 차이 있음**

---

## 🧠 보너스: ttest\_ind()로 동일 결과 확인

```python
# equal_var=True 옵션으로 합동분산 기반 검정 수행
ttest_result = stats.ttest_ind(group1, group2, equal_var=True)
print(ttest_result)  # (t-통계량, p-value)
```

---

## 📌 요약 정리

| 항목      | 결과      | 해석                  |
| ------- | ------- | ------------------- |
| F-통계량   | 약 1.348 | 분산 차이 없음 (귀무가설 채택)  |
| 합동분산    | 약 0.449 | 두 집단 통합 분산          |
| p-value | 약 0.003 | 평균 차이 유의미 (귀무가설 기각) |

---

## ✅ 실전 팁

* `var()`는 `ddof=1` 기본 포함 (`np.var(..., ddof=1)` 동일)
* **F검정 조건 주의**: 자유도가 더 큰 쪽을 분자로!
* `abs(t_stat)`와 `stats.t.cdf()` 조합은 **양측검정 핵심**
* `ttest_ind()`는 계산 검산용으로 활용 가능