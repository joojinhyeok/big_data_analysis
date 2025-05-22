## 📘 Day 9 Summary: 교차검증 & 하이퍼파라미터 튜닝

### ✅ 학습 목표

* K-Fold 교차검증을 통해 모델의 일반화 성능 평가
* `GridSearchCV`를 사용한 하이퍼파라미터 튜닝 실습
* 최적 모델로 예측 후 `submission.csv` 파일 생성

---

### 1️⃣ 교차검증 (Cross Validation)

#### ✔️ 목적

* 모델이 특정 데이터에 과적합(overfitting) 되는 것을 방지
* 데이터셋을 여러 조각으로 나누어 더 신뢰도 높은 성능 평가

#### ✔️ K-Fold 개념

* 데이터를 K개의 조각(fold)으로 나누어,
* 각 fold마다 한 번씩 테스트셋, 나머지는 훈련셋으로 사용
* 평균 정확도로 모델의 일반화 성능 평가

#### ✔️ 코드 요약

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

print("각 Fold의 정확도: ", scores)
print("평균 정확도: ", np.mean(scores))
```

---

### 2️⃣ GridSearchCV를 이용한 하이퍼파라미터 튜닝

#### ✔️ 목적

* 모델 성능에 영향을 주는 **하이퍼파라미터**를 체계적으로 탐색
* 교차검증을 기반으로 가장 성능이 좋은 조합을 선택

#### ✔️ 코드 요약

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=1
)

grid.fit(X_train, y_train)

print("최적의 파라미터:", grid.best_params_)
print("최고의 정확도:", grid.best_score_)
```

---

### 3️⃣ 예측 및 제출 파일 생성

#### ✔️ `PassengerId`를 함께 분리한 이유

* train\_test\_split 이후에도 원래 ID와 매칭하기 위해

#### ✔️ 코드 요약

```python
pred = grid.best_estimator_.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': pid_test,
    'Survived': pred
})

submission.to_csv('submission.csv', index=False)
```

---

### 🧠 오늘 배운 핵심 개념 요약

| 개념             | 설명                              |
| -------------- | ------------------------------- |
| 교차검증           | 데이터를 여러 조각으로 나누어 모델 평가의 신뢰도를 높임 |
| KFold          | 데이터를 K등분하여 반복적으로 훈련/검증          |
| 하이퍼파라미터 튜닝     | 모델 성능 향상을 위한 설정값 탐색             |
| GridSearchCV   | 가능한 하이퍼파라미터 조합을 모두 시도           |
| submission.csv | 예측 결과를 저장하여 제출 가능하게 구성          |

---

### 📝 출력 예시

```plaintext
각 Fold의 정확도:  [0.80701754 0.78947368 0.77192982 0.81578947 0.76106195]
평균 정확도:  0.7890544946436888
최적의 파라미터: {'max_depth': 6, 'min_samples_split': 2, 'n_estimators': 100}
최고의 정확도 (평균 교차검증): 0.8137090513895359
✅ submission.csv 저장 완료!