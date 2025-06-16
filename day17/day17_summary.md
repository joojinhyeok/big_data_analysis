# 📘 Day 17 Summary: 빅분기 실기 - 1유형 핵심 10문제 실습

---

## ✅ 1. 그룹별 통계

**문제**: Sex, Pclass별 생존률(Survived 평균)을 계산하고 컬럼명을 SurvivalRate로 변경, 낮은 순 정렬

```python
a = train.groupby(['Sex', 'Pclass'])['Survived'].mean().reset_index()
a = a.rename(columns={'Survived': 'SurvivalRate'})
a = a.sort_values(by='SurvivalRate', ascending=True)
````

---

## ✅ 2. 조건 필터링

**문제**: Pclass=1 이고 Sex='female'인 승객의 생존률 평균을 정수로 출력

```python
condition = (train['Pclass'] == 1) & (train['Sex'] == 'female')
a = int(train[condition]['Survived'].mean())
print(a)
```

---

## ✅ 3. 정렬 후 추출

**문제**: Fare가 가장 높은 5명의 Name과 Fare를 내림차순으로 출력

```python
sort_train = train.sort_values(by='Fare', ascending=False)
top5 = sort_train[['Name', 'Fare']].head(5)
print(top5)
```

---

## ✅ 4. 범주형 변수 처리

**문제**: Sex 컬럼을 male → 0, female → 1로 변환 후 고유값 출력

```python
train['Sex'] = train['Sex'].replace({'male': 0, 'female': 1})
print(train['Sex'].unique())
```

---

## ✅ 5. 파생변수 생성

**문제**: Age ≤ 15 → 'child', 나머지 → 'Adult'로 AgeGroup 컬럼 생성, 고유값 출력

```python
train['AgeGroup'] = np.where(train['Age'] <= 15, 'child', 'Adult')
print(train['AgeGroup'].unique())
```

---

## ✅ 6. 조건부 평균 계산

**문제**: Fare ≥ 30인 승객의 평균 Age를 정수로 출력 (반올림)

```python
condition = train['Fare'] >= 30
mean_age = int(round(train[condition]['Age'].mean()))
print(mean_age)
```

---

## ✅ 7. 다중 조건 + 그룹 통계

**문제**: Embarked='S'이고, Sex='male'인 승객의 Pclass별 생존률 계산

```python
condition = (train['Embarked'] == 'S') & (train['Sex'] == 'male')
filtered = train[condition]
ps = filtered.groupby('Pclass')['Survived'].mean().reset_index()
ps = ps.rename(columns={'Survived': 'SurvivalRate'})
ps = ps.sort_values(by='Pclass')
print(ps)
```

---

## ✅ 8. 결측치 수 및 비율 계산

**문제**: Age 컬럼의 결측치 개수 및 전체 대비 비율(%) 계산 (정수, 반올림)

```python
a = train['Age'].isnull().sum()
b = round(a / len(train) * 100)
print(a)
print(b)
```

---

## ✅ 9. 이상치 탐지 및 처리 (IQR)

**문제**: Fare 컬럼의 이상치를 중앙값으로 대체 (IQR 기준)

```python
Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

condition = (train['Fare'] < lower) | (train['Fare'] > upper)
median_fare = train['Fare'].median()
train.loc[condition, 'Fare'] = median_fare
```

---

## ✅ 10. apply() + 사용자 정의 함수

**문제**: Name 컬럼에서 Mr/Mrs 등 호칭만 추출해 Title 컬럼 생성, 고유값 출력

```python
def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()

train['Title'] = train['Name'].apply(extract_title)
print(train['Title'].unique())
```

---

## 🎯 총정리

* `.groupby()`, `.mean()`, `.sort_values()`, `.reset_index()` → 그룹통계
* `isnull().sum()`, `len(df)` → 결측치 수 & 비율
* `np.where()`, `apply()` → 파생변수 생성
* `loc[]` → 조건 필터링 및 수정
* `quantile()`, IQR → 이상치 처리
* 실전에서도 가장 자주 등장하는 1유형 핵심 패턴 10개 마스터!