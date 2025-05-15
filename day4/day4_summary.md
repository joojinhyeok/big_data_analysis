# 📅 Day 4 Summary – 타이타닉 생존 예측 모델링

## ✅ 목표
- 분류(Classification) 문제의 흐름을 이해하고,
- 타이타닉 데이터를 기반으로 로지스틱 회귀 모델을 학습하고 평가함
- 실제 test 데이터에 대해 예측 결과를 CSV로 저장함

---

## 🔍 사용한 Feature
- `Pclass` (객실 등급)
- `Age` (나이)
- `Fare` (요금)
- `Sex_male` (성별 인코딩된 값)

---

## ⚙️ 학습 절차

1. `train_test_split`으로 학습/검증 데이터 분리 (`test_size=0.2`)
2. `LogisticRegression` 모델 학습 (`max_iter=1000`)
3. 검증 데이터에 대한 예측 수행
4. 정확도 평가 → **Validation Accuracy: 75.48%**

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy_score(y_val, y_pred)
