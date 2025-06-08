# 📅 Day 15 Summary – 2유형 문제 풀이 정리

오늘은 빅데이터분석기사 실기 2유형 유형 2문제를 풀었고,  
실전 문제에 맞춰 전처리, 모델 학습, 예측, 결과 저장까지 수행함.

---

## ✅ 문제 1: 타이타닉 생존자 예측 (기출 변형)

### 📌 문제 유형

- `train.csv`, `test.csv` 사용
- 생존 여부(`Survived`)를 예측하여 `submission.csv`로 저장

### 🧩 주요 지시사항

1. 불필요한 컬럼 제거: `'PassengerId', 'Name', 'Ticket', 'Cabin'`
2. `train`: 결측치 제거  
3. `test`:  
   - 수치형(`Age`, `Fare`) → `median()`  
   - 범주형(`Embarked`) → `'S'`
4. `LabelEncoder`: `Sex`, `Embarked` 컬럼 인코딩
5. 모델 3종 학습 + 교차검증 (`cv=5`)
6. `StackingClassifier`로 예측
7. `submission.csv`로 저장 (`PassengerId`, `Survived`, `index=False`)

---

### ⚙️ 모델 구성

- Base: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`
- Meta: `LogisticRegression`

---

## ✅ 문제 2: 고객 구매 분석 (실전형 2유형)

### 📌 문제 유형

- `train_A.csv`, `test_A.csv` 사용
- `Purchased` 여부 예측 → `CustomerID`, `Purchased` 컬럼으로 `submission.csv` 저장

### 🧩 주요 지시사항

1. 불필요한 컬럼 제거: `'CustomerID', 'Name', 'PhoneNumber'`
2. `train`: 결측치 제거
3. `test`:  
   - 수치형(`Age`, `Income`) → `median()`  
   - 범주형(`Region`) → `mode()`
4. `LabelEncoder`: `Gender`, `Region` → `pd.concat(train+test)`로 fit 후 transform
5. 모델 3종 학습 + 5-폴드 교차검증
6. `StackingClassifier`로 예측
7. `submission.csv` 저장 (index=False)

---

### ⚙️ 사용 모델

- Base: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`
- Meta: `LogisticRegression`

---

## 🧠 실전에서 유의할 점

- `LabelEncoder`는 반드시 `train+test` 전체를 기준으로 fit해야 unseen label 오류 방지
- `cross_val_score()`의 `cv`는 클래스 분포를 고려해야 하며, 최소 샘플 수 확인 필수
- `submission.csv` 저장 시 반드시 `index=False`, 컬럼 순서도 지시에 맞춰야 함

---

## 🗃️ 생성된 파일

- `submission.csv`: 타이타닉 생존 예측 결과
- `submission_2.csv`: 고객 구매 예측 결과