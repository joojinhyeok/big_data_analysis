# 📅 Day 8 – 모델 예측 및 제출파일 생성 실습

이번 실습에서는 학습된 모델을 활용해 테스트 데이터를 예측하고, `.csv` 제출 파일을 만드는 전체 흐름을 실습하였습니다.

---

## ✅ 실습 목표

- 학습된 모델(.pkl) 불러오기
- 테스트 데이터 전처리
- 컬럼 순서 맞추기 (reindex)
- 예측 결과를 CSV 파일로 저장하여 제출 형식 갖추기

---

## 🧪 실습 단계 요약

### 1단계: 테스트 데이터 불러오기 + 전처리

- `test.csv` 파일 불러오기
- 결측치 제거 (`Age`, `Fare`, `Embarked` 등)
- `PassengerId` 따로 저장 (제출용 ID)
- `get_dummies()`로 범주형 변수 인코딩

### 2단계: 모델과 컬럼 정보 불러오기

- `joblib.load()`로 학습된 모델(`model.pkl`) 불러오기
- `model_columns.pkl`로 학습 당시 사용한 컬럼 순서 불러오기

### 3단계: 컬럼 순서 맞추기 (reindex)

- `X_test = X_test.reindex(columns=expected_columns, fill_value=0)`
- 없는 컬럼은 0으로 채우고 순서 맞춤

### 4단계: 예측 수행 및 제출 파일 생성

- `model.predict(X_test)`로 예측 수행
- `submission = pd.DataFrame(...)` 생성
- `to_csv('submission.csv', index=False)`로 파일 저장

---

## 📁 생성 파일 요약

| 파일명 | 설명 |
|--------|------|
| `model.pkl` | 학습된 모델 |
| `model_columns.pkl` | 모델 학습 당시 사용한 컬럼 순서 |
| `submission.csv` | 최종 예측 결과 제출용 파일 |