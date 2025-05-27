# 📅 Day 7 – 실전 모의고사 실습 및 전체 흐름 정리

## ✅ 목표
- 실전 스타일 데이터셋으로 모델 학습 전체 흐름 실습
- 학습 → 저장 → 예측 → 제출까지 모든 과정 연습
- 실기 시험 대비 코드 템플릿 확보

---

## 📁 사용 데이터

- `train.csv` : 학습용 타이타닉 데이터
- `test.csv` : 예측용 테스트 데이터

---

## 🧪 전처리 요약

| 단계  | 내용         | 컬럼         | 처리 방식                              |
|-------|--------------|--------------|----------------------------------------|
| 1단계 | 결측치 처리    | Age, Embarked, Fare | 평균, 최빈값으로 채움                    |
| 2단계 | 불필요한 컬럼 제거 | Name, Ticket, Cabin | 정보 부족 / 의미 없음으로 제거           |
| 3단계 | 인코딩       | Sex, Embarked, Pclass | 원-핫 인코딩 (get_dummies, drop_first=True) |
| 4단계 | 컬럼 정렬     | train/test 통일  | reindex(columns=...) 사용                |

---

## 🧠 모델링 및 평가

| 항목       | 내용                              |
|------------|-----------------------------------|
| 모델       | RandomForestClassifier (n=100)   |
| 훈련/검증 분리 | train_test_split (test_size=0.2) |
| 평가 지표   | accuracy_score                    |
| 검증 정확도 | 0.8212                            |

---

## 💾 모델 저장

- `best_model.pkl` : 학습된 모델 저장
- `model_columns.pkl` : 학습 시 사용한 컬럼 순서 저장 (test셋 적용 시 필요)

```python
joblib.dump(model, 'best_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
