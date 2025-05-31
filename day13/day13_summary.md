# 📘 Day 13 Summary: 실전 시험 대비 - 스태킹 앙상블 + 성능 향상

## ✅ 학습 목표
- 실기 시험에서 출제될 수 있는 앙상블 문제에 대응
- 전처리 → 모델 정의 → 하이퍼파라미터 튜닝 → 결과 제출 흐름 익히기
- StackingClassifier를 통한 성능 향상 전략 이해 및 적용

---

## 🧪 실습 문제 (모의시험 스타일)

**Titanic 생존자 예측 문제를 해결하라.**  
다양한 모델을 조합한 스태킹 앙상블 기법을 활용하여 성능을 높이고, 최종 제출 파일을 생성하시오.

### 🔹 조건
1. 주어진 `train.csv`, `test.csv`, `submission.csv` 파일을 사용한다.
2. 스태킹 기반 앙상블 모델을 사용하여 예측 정확도를 최대화할 것.
3. 성능 향상을 위해 전처리, 스케일링, 모델 구성 또는 하이퍼파라미터 튜닝 등을 시도할 것.
4. 최종 예측 결과를 `final_submission.csv` 파일로 저장한다.

### 🔹 요구사항
- 전처리: 결측치 처리, 불필요한 열 제거, 범주형 인코딩
- 모델링: 최소 3개 이상의 기반 모델 + 최종 예측 모델 구성
- 성능 비교: 변경 실험에 따른 정확도 비교
- 결과 제출: `submission.csv`, `final_submission.csv` 생성

---

## ✅ 전처리 요약
- **열 제거**: `PassengerId`, `Name`, `Ticket`, `Cabin`
- **결측치 처리**: 
  - `train` → dropna()
  - `test` → 수치형 median / `Embarked`는 'S'로 채움
- **Label Encoding**: `Sex`, `Embarked`

---

## ✅ 모델 구성
| 모델 이름             | 설명                                             |
|----------------------|--------------------------------------------------|
| DecisionTree         | 조건 기반의 트리 분류기                           |
| LogisticRegression   | 선형 회귀 기반 이진 분류기                        |
| KNeighborsClassifier | 거리 기반 분류기 (k-NN)                          |
| RandomForest         | 여러 트리를 앙상블한 모델                         |
| StackingClassifier   | 여러 모델을 조합하여 최종 예측기와 결합한 모델   |

---

## ✅ 단계별 실습 결과

### 📌 기본 성능 비교
| 모델                 | 정확도         |
|----------------------|----------------|
| DecisionTree         | 0.706          |
| LogisticRegression   | 0.797          |
| KNeighborsClassifier | 0.685          |
| RandomForest         | 0.790          |
| **Stacking**         | **0.804**      |

---

### 📌 스태킹 모델 성능 향상 시도

#### ✅ 실험 1: KNN 제거 → 단순 스태킹 (`dt`, `lr`)
- 정확도: **0.790**

#### ✅ 실험 2: 최종 예측기 → LogisticRegression 교체
- 정확도: **0.811**
- ✔️ 더 단순한 모델이 성능이 더 높게 나올 수도 있음 (과적합 방지)

---

## ✅ 스케일링 적용 실험

### 🔹 사용 도구
- `StandardScaler`: 평균 0, 표준편차 1로 정규화

### 🔹 이유
- KNN, LogisticRegression은 스케일에 민감함
- Tree 기반 모델은 영향 없음

| 조건                 | 정확도         |
|----------------------|----------------|
| 스케일링 후 stacking | 0.811          |

---

## ✅ GridSearchCV: 하이퍼파라미터 튜닝

### 🔹 대상 모델: RandomForest (최종 예측기)
- `max_depth`: [3, 5, 10]
- `n_estimators`: [50, 100, 150]

### 🔹 결과
- **Best Params**: `max_depth=5`, `n_estimators=50`
- **Best Score (cv)**: 0.8119

---

## ✅ 최종 결과

- 최종 스태킹 정확도: **0.832**
- `final_submission.csv` 생성 완료
- 전체 학습 데이터를 다시 학습하여 예측한 결과 사용