# 📘 Day 10 Summary - Titanic ML Pipeline

## 🗂️ 목표
- 데이터 전처리 → 모델 학습 → 하이퍼파라미터 튜닝 → 예측 → 제출까지 전체 흐름 구현
- 실기 시험을 위한 실전 파이프라인 연습

---

## 1. 데이터 불러오기
```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
````

---

## 2. 결측치 처리

```python
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

train.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True, errors='ignore')
test.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True, errors='ignore')
```

---

## 3. 범주형 변수 인코딩

```python
train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)
```

---

## 4. 데이터 분할

```python
X_train = train.drop(columns=['Survived', 'PassengerId'])
y_train = train['Survived']
pid_test = test['PassengerId']
X_test = test.drop(columns=['PassengerId'])

X_train_split, X_valid, y_train_split, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)
```

---

## 5. 모델 정의 및 학습

```python
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train_split, y_train_split)
```

---

## 6. 하이퍼파라미터 튜닝

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=1)
grid.fit(X_train_split, y_train_split)
```

---

## 7. 예측 및 제출

```python
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
pred = grid.best_estimator_.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': pid_test,
    'Survived': pred
})

submission.to_csv('submission.csv', index=False)
```

---

## ✅ 사용된 개념 요약

* `fillna()` – 결측값 대체
* `get_dummies()` – 범주형 변수 인코딩
* `train_test_split()` – 훈련/검증 분할
* `RandomForestClassifier` – 앙상블 모델
* `GridSearchCV` – 하이퍼파라미터 튜닝
* `reindex()` – 컬럼 정렬
* `.to_csv()` – 결과 저장