# 📘 Day 19 Summary: Titanic 생존 예측 (2유형 분류 문제)

## ✅ 문제 개요
- Titanic 탑승객 데이터를 활용하여 생존 여부(`Survived`)를 예측하는 분류 모델 생성
- 모델: `RandomForestClassifier`
- 평가 지표: `accuracy_score`
- 제출 형식: `result.csv` (컬럼: `pred`)

---

## 🧭 2유형 실습 흐름

### 1. 데이터 확인
- `train.csv`, `test.csv` 불러오기
- 수치형 결측치: `Age`, `Fare`
- 범주형 결측치: `Embarked`

### 2. 데이터 전처리
- **제거 컬럼**: `'PassengerId', 'Name', 'Ticket', 'Cabin'`
- **결측치 처리**
  - `Age`: 평균값
  - `Embarked`: 최빈값
  - `Fare` (test): 평균값
- **범주형 인코딩**
  - `Sex`, `Embarked`: `LabelEncoder` 사용

### 3. 데이터 분할
- 독립변수 `X` = `train.drop('Survived', axis=1)`
- 종속변수 `y` = `train['Survived']`
- `train_test_split()`으로 훈련/검증 데이터 분리  
  - `test_size=0.3`, `random_state=42`

### 4. 모델 학습
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)
````

### 5. 성능 평가

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_val, pred1)
```

### 6. 예측 및 제출

* test 데이터 예측
* 예측 결과를 `result.csv`로 저장

```python
pd.DataFrame({'pred': pred2}).to_csv('result.csv', index=False)
```

---

## 🧪 주요 실수 체크

* `drop()` 사용 시 `inplace=True` 또는 재할당 필요
* `test['Survived']`는 존재하지 않음 (y 분리 시 주의)
* test 예측 시 `model.predict(test)` ← 그대로 사용해도 됨 (drop 후라면)

---

## ✅ 결과 파일 예시 (`result.csv`)

| pred |
| ---- |
| 0    |
| 1    |
| ...  |

---

## 📌 오늘 배운 핵심

| 내용                    | 요약                    |
| --------------------- | --------------------- |
| LabelEncoder 순서       | 결측치 먼저 처리 → 인코딩       |
| train/test 분할 후 평가 지표 | `accuracy_score` 사용   |
| 실기 스타일 작성법            | 주석 기반 요구사항 분석 → 코드 구현 |
| 제출 양식                 | 컬럼명 `pred`, 인덱스 없이 저장 |