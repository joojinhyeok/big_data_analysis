# 📅 Day 5 – 모델 성능 향상 및 제출 파일 생성

## ✅ 오늘의 학습 목표

- 로지스틱 회귀(Logistic Regression)를 활용한 분류 문제 해결
- `GridSearchCV`를 통한 하이퍼파라미터 튜닝
- 학습된 모델 저장 및 재사용
- 테스트 데이터 예측
- `submission.csv` 제출 파일 생성

---

## 📁 사용 데이터

- `train.csv`: 학습용 타이타닉 데이터
- `test.csv`: 예측용 테스트 데이터

---

## 🧪 실습 흐름 요약

### 1. 학습 데이터 불러오기 및 전처리

```python
train = pd.read_csv('C:/csv/train.csv')
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
train = train.dropna(subset=features)
X = pd.get_dummies(train[features], drop_first=True)
y = train['Survived']
```

### 2. 데이터 분리 및 모델 학습

```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
params = {'C': [0.01, 0.1, 1, 10], 'max_iter': [100, 500, 1000]}
grid = GridSearchCV(LogisticRegression(), param_grid=params, cv=5)
grid.fit(X_train, y_train)
```

### 3. 성능 확인

```python
model = grid.best_estimator_
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
```

### 4. 모델 및 피처 정보 저장

```python
import joblib
joblib.dump(model, 'best_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
```

---

## 📤 테스트 예측 및 제출 파일 생성 (exam02.py)

### 1. 테스트 데이터 불러오기 및 전처리

```python
test = pd.read_csv('C:/csv/test.csv')
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])
X_test = pd.get_dummies(test[features], drop_first=True)
```

### 2. 저장된 모델 및 피처 순서 불러오기

```python
model = joblib.load('C:/python/big_data_analysis/best_model.pkl')
expected_columns = joblib.load('C:/python/big_data_analysis/model_columns.pkl')
X_test = X_test.reindex(columns=expected_columns, fill_value=0)
```

### 3. 예측 및 파일 생성

```python
predictions = model.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.to_csv('C:/csv/submission.csv', index=False)
print("submission.csv 저장 완료!")
```

---

## 🧠 오늘 배운 핵심 개념 정리

| 개념                       | 설명                                         |
| -------------------------- | -------------------------------------------- |
| `LogisticRegression`       | 이진 분류에 사용되는 머신러닝 모델           |
| `GridSearchCV`             | 여러 파라미터 조합을 시험해 최적 모델 탐색   |
| `joblib.dump()` / `load()` | 모델 저장 및 불러오기                        |
| `get_dummies()`            | 범주형 데이터를 숫자로 변환 (원-핫 인코딩)   |
| `reindex()`                | 테스트 데이터의 열 순서를 학습 데이터와 맞춤 |

---

## 💡 문제 해결 히스토리

- ✅ `GridSearchCV`로 모델 학습 후 `.pkl` 저장 성공
- ⚠️ `UnicodeEncodeError`: 이모지(✅) → 일반 문자열로 교체하여 해결
- ⚠️ `FileNotFoundError`: 모델 경로를 절대경로로 수정하여 해결
- ✅ 최종적으로 `submission.csv` 생성 성공

---

## 🎯 다음 단계 예고 (Day 6)

- 앙상블 모델 학습 (Voting, Bagging, Boosting)
- 다양한 분류 모델 비교 (Decision Tree, RandomForest, XGBoost 등)

```

```
