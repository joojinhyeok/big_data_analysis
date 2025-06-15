# 📘 Day 16 Summary: 1유형 Titanic 데이터 전처리 실습

## ✅ 사용 데이터셋
- Titanic 생존자 예측 데이터 (`train.csv`)
- 주요 컬럼:
  - `Age`: 나이
  - `Fare`: 운임
  - `Embarked`: 승선 항구
  - `Pclass`: 객실 등급

---

## 🧪 1. Age 컬럼 결측치 처리

### 📌 문제
> `Age` 컬럼에 결측치가 존재한다.  
> `Pclass`별 평균 나이로 결측치를 채우시오.

### ✅ 코드
```python
train['Age'] = train['Age'].fillna(train.groupby('Pclass')['Age'].transform('mean'))
````

### ✅ 설명

* `groupby('Pclass')`: 객실 등급(Pclass) 기준으로 그룹화
* `.transform('mean')`: 각 Pclass 그룹의 평균 나이 생성
* `.fillna(...)`: 해당 평균 나이로 결측치 대체

---

## 🧪 2. Fare 컬럼 이상치 처리

### 📌 문제

> `Fare` 컬럼의 이상치를 중앙값으로 대체하시오.
> 이상치 기준:
>
> * 1사분위수(Q1)보다 1.5 \* IQR 아래
> * 3사분위수(Q3)보다 1.5 \* IQR 위

### ✅ 코드

```python
Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

median_fare = train['Fare'].median()

train.loc[(train['Fare'] < lower_bound) | (train['Fare'] > upper_bound), 'Fare'] = median_fare
```

### ✅ 설명

* `quantile()`로 Q1, Q3 계산 → IQR 산출
* 이상치 조건 계산 후, 해당 조건을 만족하는 값은 중앙값으로 대체

---

## 🧪 3. Embarked 컬럼 결측치 처리

### 📌 문제

> `Embarked` 컬럼에 결측치가 존재한다.
> 최빈값(가장 많이 나타난 값)으로 결측치를 채우시오.

### ✅ 코드

```python
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
```

### ✅ 설명

* `mode()`는 최빈값 반환 → `[0]`으로 첫 번째 최빈값 선택
* `fillna()`로 결측치 대체

---

## ✅ 출력 확인 예시

```python
print(train['Embarked'])
print(train['Embarked'].isnull().sum())  # 결측치 없는지 확인용
```

---

## 📌 마무리

* 1유형은 결측치 처리, 이상치 처리, 그룹별 평균 등 전처리 기초를 잘 묻는 유형
* 핵심 메서드: `fillna()`, `groupby()`, `transform()`, `quantile()`, `median()`, `mode()`, `loc[]`