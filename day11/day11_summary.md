# 📚 Day 11 - 앙상블 기법 실습 정리 (Voting / Bagging / Stacking)

## ✅ 실습 개요

- Titanic 생존 예측 데이터를 기반으로 앙상블 모델 학습
- 각 앙상블 방식별 정확도 및 구조 이해

---

## 1️⃣ VotingClassifier

### 🧠 개념
- 여러 모델이 각자 예측하고, **투표 방식으로 최종 결정**
- **Hard Voting**: 다수결
- **Soft Voting**: 예측 확률 평균

### ✅ 사용 모델
- LogisticRegression
- DecisionTreeClassifier
- KNeighborsClassifier

### 💡 결과
- Hard Voting Accuracy: 0.7933
- Soft Voting Accuracy: 0.8212 ✅

---

## 2️⃣ BaggingClassifier vs RandomForestClassifier

### 🧠 개념
| 모델 | 설명 |
|------|------|
| BaggingClassifier | 샘플만 랜덤하게 뽑아 여러 모델 학습 |
| RandomForestClassifier | 샘플 + 피처 모두 랜덤하게 뽑아 다양한 결정트리 학습 |

### ✅ 정확도 비교
- BaggingClassifier Accuracy: 0.8045
- RandomForestClassifier Accuracy: 0.8268 ✅

---

## 3️⃣ StackingClassifier

### 🧠 개념
- 여러 모델(base_estimators)이 예측한 결과를 **최종 메타 모델**이 다시 예측
- 보통 **복잡한 문제에 효과적**

### ✅ 구성
- Base models: LogisticRegression, DecisionTree, KNN
- Final model: RandomForest

### ❗ 결과
- Stacking Accuracy: 0.6983 (예상보다 낮음)

### 🔍 원인 추정
- base 모델 조합의 한계
- 피처 스케일링 미적용 (특히 KNN)
- 데이터 양 부족 시 stacking은 불안정

---

## 🔁 앙상블 모델 비교 요약

| 모델 | 설명 | 정확도 |
|------|------|--------|
| Voting (Soft) | 확률 기반 투표 | 0.8212 |
| Bagging | 샘플 랜덤 트리 | 0.8045 |
| RandomForest | 샘플 + 피처 랜덤 트리 | 0.8268 |
| Stacking | base 모델 조합 + meta 예측 | 0.6983 |

---

## 💡 오늘의 핵심 정리

- **Soft Voting > Hard Voting**: 확률 기반이 성능 좋음
- **RandomForest가 Bagging보다 다양성, 정확도 우수**
- **Stacking은 고급 기술이지만 모델 조합, 데이터 분포에 따라 성능 변동 큼**
- KNN 등 거리 기반 모델 사용 시에는 **스케일링 필수!**

---

## 📌 실전 꿀팁

- 실기 시험에서는 Voting, RandomForest 자주 출제됨
- Stacking은 파이프라인 개념에 익숙해져야 함
- 모델 정확도 비교 후 적절한 방식 선택할 것
