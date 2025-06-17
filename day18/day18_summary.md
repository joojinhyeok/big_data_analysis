# 📘 Day 18 Summary: 빅데이터분석기사 실기 체험환경 2유형 실습

## ✅ 실습 개요

- **문제 유형**: 2유형 (회귀 문제)
- **목표**: 고객의 '총구매액' 예측
- **데이터**: `customer_train.csv`, `customer_test.csv`
- **모델**: `RandomForestRegressor` 사용

---

## 🔍 실습 흐름 정리

### 1️⃣ 데이터 확인

- `train`과 `test` 데이터셋에서 주요 컬럼 확인
- 결측치 존재: `환불금액`
- 범주형 컬럼: `주구매상품`, `주구매지점`

### 2️⃣ 데이터 전처리

- **결측치 처리**
  - `환불금액` → **중앙값(`median`)**으로 대체

- **범주형 인코딩**
  - `LabelEncoder` 사용하여 `주구매상품`, `주구매지점`을 수치형으로 변환

### 3️⃣ 데이터 분할

- 종속변수: `총구매액`
- 독립변수: `회원ID`, `총구매액` 제외한 나머지
- `train_test_split` (40% 테스트 비율, `random_state=42` 고정)

### 4️⃣ 모델링 및 학습

- 모델: `RandomForestRegressor`
  - `n_estimators=200`, `max_depth=10`, `random_state=42`
- 학습: `.fit(X_train, y_train)`

### 5️⃣ 성능 평가

- **지표**: `RMSE (Root Mean Squared Error)`
  - 계산: `root_mean_squared_error(y_test, pred1)`
  - RMSE 값 출력

### 6️⃣ 테스트셋 예측 및 결과 저장

- `test` 데이터 전처리 후 예측 수행
- 결과를 `result.csv`로 저장
  ```python
  pd.DataFrame({'pred': pred2}).to_csv('result.csv', index=False)
````

---

## 🧠 주요 개념 정리

| 항목                          | 설명                          |
| --------------------------- | --------------------------- |
| `RandomForestRegressor`     | 회귀 문제 해결을 위한 앙상블 모델 (트리 기반) |
| `LabelEncoder`              | 범주형 변수를 숫자형으로 변환            |
| `fillna(median)`            | 수치형 결측치를 중앙값으로 대체           |
| `train_test_split()`        | 데이터 분할 함수                   |
| `root_mean_squared_error()` | 예측 성능 측정 지표 (값이 낮을수록 좋음)    |

---

## 🗂️ 생성 파일

* `result.csv` : 테스트 데이터셋 예측 결과 파일

---

## 💡 느낀 점 / 팁

* 결측치 처리 방식(평균, 중앙값, 최빈값)에 따라 성능 차이 확인 가능
* `LabelEncoder`는 `train` 기준으로 fit 후, `test`에 그대로 transform 해야 함 (데이터 유출 방지)
* 회귀 문제에서는 `RandomForestRegressor`를, 분류 문제에서는 `RandomForestClassifier`를 사용