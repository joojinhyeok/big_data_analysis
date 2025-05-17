# 📅 Day 6 – 모델 성능 개선 및 앙상블 기법

## ✅ 오늘의 학습 목표
- 다양한 머신러닝 분류 모델 성능 비교
- 교차검증(`cross_val_score`)을 통한 모델 신뢰도 평가
- 앙상블 기법(Voting, Stacking)을 통한 성능 향상 실습
- 최종 모델 선택 및 `submission.csv` 생성

---

## 🧪 실습 내용 요약

### 1. 다양한 분류 모델 성능 비교 (`model_compare.py`)
| 모델 | 검증 정확도 (`X_val`) |
|------|------------------------|
| Logistic Regression | 0.7972 |
| Random Forest        | 0.7622 |
| KNN                  | 0.6643 |
| SVM                  | 0.6364 |
| XGBoost              | 0.7832 |

---

### 2. 교차 검증 결과 (`cross_val_score`, 5-Fold 기준)
| 모델 | 평균 정확도 | 표준편차 |
|------|-------------|-----------|
| Logistic Regression | 0.7823 | 0.0337 |
| Random Forest        | 0.7852 | 0.0366 |
| KNN                  | 0.6644 | 0.0509 |
| SVM                  | 0.6630 | 0.0707 |
| XGBoost              | **0.7894** | **0.0290** ✅

---

### 3. 앙상블 모델 성능 (`ensemble_models.py`)
| 앙상블 방식     | 정확도 |
|----------------|--------|
| Voting (Hard)  | **0.8252** ✅
| Voting (Soft)  | 0.8182 |
| Stacking       | 0.7832 |

- `Voting (Hard)`가 가장 우수한 성능을 보여 최종 선택됨

---

### 4. 최종 예측 및 제출 파일 생성 (`generate_submission.py`)
- `VotingClassifier`를 전체 학습 데이터로 재학습
- `test.csv` 전처리 후 예측 수행
- `submission.csv` 파일 생성 완료 🎉

---

## 🧠 오늘의 인사이트
- 단일 모델 성능보다 **앙상블 모델이 전반적으로 우수함**
- 단순한 `Voting(Hard)` 조합도 성능 향상에 크게 기여할 수 있음
- `cross_val_score`는 모델 신뢰도 판단에 필수적임
- 피처 일치(`model_columns.pkl`)가 test 예측 시 중요함

---

## 📁 생성된 주요 파일
- `model_compare.py` – 분류 모델 성능 비교
- `ensemble_models.py` – 앙상블 기법 실습
- `generate_submission.py` – 예측 및 제출 파일 생성
- `submission.csv` – 최종 제출 파일
