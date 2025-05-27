# 📚 Day 12 - Stacking & 하이퍼파라미터 튜닝 실습 (시험 대비)

## ✅ 실습 개요

- Titanic 생존 예측 데이터를 기반으로 스태킹 앙상블 학습
- GridSearchCV 및 RandomizedSearchCV로 하이퍼파라미터 최적화
- 시험 스타일로 전체 흐름 구성: 전처리 → 모델링 → 평가

---

## 1️⃣ 기본 StackingClassifier 구성

- 사용 모델:
  - DecisionTreeClassifier
  - KNeighborsClassifier
  - 최종 메타 모델: LogisticRegression

- 정확도 결과:
  - **StackingClassifier Accuracy: 0.7482**

---

## 2️⃣ GridSearchCV 튜닝

- 최적 파라미터:
  - `max_depth: 5`
  - `min_samples_split: 2`

- 튜닝 후 정확도:
  - **Tuned StackingClassifier Accuracy: 0.7413**

---

## 3️⃣ 교차검증 평가

- Fold별 정확도:
  - `[0.7692, 0.8252, 0.8042, 0.7832, 0.8099]`

- 평균 정확도:
  - **0.7983**

---

## 4️⃣ RandomizedSearchCV 튜닝

- 탐색 범위:
  - `max_depth`: randint(3, 10)
  - `min_samples_split`: randint(2, 10)

- 최적 조합:
  - `{'max_depth': 5, 'min_samples_split': 9}`

- 정확도:
  - **Tuned StackingClassifier Accuracy (Randomized): 0.7343**

---

## 5️⃣ 최종 시험 스타일 종합 예제

- 전처리: dropna(), LabelEncoder
- 모델 구성: StackingClassifier (튜닝된 DecisionTree + KNN)
- 평가 지표: accuracy, cross_val_score
- 실기 시험 제출 가능한 구조로 정리

---

## 📝 정리 및 배운 점

- StackingClassifier는 다양한 모델을 결합해 성능을 높일 수 있는 앙상블 방법
- GridSearchCV는 정확한 탐색, RandomizedSearchCV는 빠른 탐색에 적합
- 튜닝이 항상 성능 향상을 보장하진 않으며, 교차검증으로 안정성 평가가 중요
