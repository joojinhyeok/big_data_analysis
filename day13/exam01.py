# Day 13 실기 연습 - 스태킹 & 모델 튜닝

# 1. 라이브러리 import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 2. 데이터 불러오기
train = pd.read_csv('C:/csv/train.csv')
test = pd.read_csv('C:/csv/test.csv')
submission = pd.read_csv('C:/csv/submission.csv')


# 3. 전처리 - 불필요한 열 제거
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train = train.drop(columns=drop_cols)
test_passenger_ids = test['PassengerId']    # 나중에 submission에 사용
test = test.drop(columns=drop_cols)

# 4. 결측치 처리
train = train.dropna()
# test 데이터 프레임에서 수치형 열만 골라서 각 열의 중앙값(median)을 계산한다.
test = test.fillna(test.median(numeric_only=True))  # 수치형 결측치 보완
# Embarked(항구) 데이터에서 가장 많이 등장한 'S'로 채움(최빈값)
test['Embarked'] = test['Embarked'].fillna('S')  # 범주형 Embarked 결측치

# 5. 범주형 라벨 인코딩 - "문자(범주형)"로 되어 있는 Sex와 Embarked 컬럼을 
#                       숫자로 바꾸는 작업
le = LabelEncoder() # LabelEncoder() : 문자형(범주형) 데이터를 숫자로 변환
for col in ['Sex', 'Embarked']:
    # fit_transform()
    # fit: train[col]에서 어떤 값들이 있는지 학습 - ex) male=1 / female=0 으로
    # transform: 학습한 매핑대로, train[col]의 값을 실제 숫자로 변환
    # ex) train['Sex'] = ['male', 'female', 'male']
    #    -> le.fit_transform(train['Sex']) : [1, 0, 1]
    train[col] = le.fit_transform(train[col])
    # 주의: train 기준으로 transform
    test[col] = le.transform(test[col]) 

# 6. 학습/검증 데이터 분리
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 42)

# 7. 모델 정의
# 결정 트리: 조건을 따라 나무처럼 분류. 직관적이고 해석이 쉽지만 과적합 위험 존재 
dt = DecisionTreeClassifier(random_state=42)

# 로지스틱 회귀: 확률 기반의 이진 분류. 속도 빠르고 선형 문제에 강함. 복잡한 문제에는 x
lr = LogisticRegression(max_iter=1000)

# K-최근접 이웃: 가까운 이웃을 보고 분류. 직관적이고 학습 빠름. 느린 예측과 차원 민감 
knn = KNeighborsClassifier()

# 랜덤 포레스트: 여러 트리 앙상블. 강력한 성능과 과적합 방지. 느리고 해석이 어려움
rf = RandomForestClassifier(n_estimators=100, random_state=42)

stacking = StackingClassifier(
    estimators=[('dt', dt), ('lr', lr), ('knn', knn)],
    final_estimator=rf,
    cv=5    # K-Fole 교차검증 방식. 훈련 데이터를 5개로 나눠서 기반 모델들의 예측 결과 생성
)

# 8. 모델 학습 및 평가
for model in [dt, lr, knn, rf, stacking]:
    model.fit(X_train, y_train) # 모델 학습
    pred = model.predict(X_val) # 예측
    # model.__class__.__name__: 파이썬 객체의 "클래스 이름(문자열)"을 뽑아주는 코드
    print(model.__class__.__name__, "정확도", accuracy_score(y_val, pred))

# 9. 최종 예측 & 제출 파일 생성
stacking.fit(X, y)  # 전체 학습 데이터로 다시 학습
final_pred = stacking.predict(test)

submission['Survived'] = final_pred
submission['PassengerId'] = test_passenger_ids
submission.to_csv('submission.csv 저장 완료!')

# --------------------------------------------------------------
# 2단계 실습
# 스태킹 모델을 기반으로 성능 향상 시도
# 하이퍼파라미터 튜닝 or 구성 모델 바꾸기 or 전처리 다르게 해보기
# --------------------------------------------------------------
# (1) knn 제거, 단순 스태킹
# 기반 모델: dt, lr (2개)
stacking_v1 = StackingClassifier(
    estimators=[('dt', dt), ('lr', lr)],
    final_estimator=rf,
    cv=5
)
stacking_v1.fit(X_train, y_train)
pred_v1 = stacking_v1.predict(X_val)
print("stacking_v1 정확도", accuracy_score(y_val, pred_v1))

# (2) final_estimator를 LogisticRegression으로 교체
# 기반 모델: dt, lr, knn / 최종 모델: lr
stacking_v2 = StackingClassifier(
    estimators=[('dt', dt), ('lr', lr), ('knn', knn)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
stacking_v2.fit(X_train, y_train)
pred_v2 = stacking_v2.predict(X_val)
print("stacking_v2 정확도", accuracy_score(y_val, pred_v2))

# ** 결과 **
# Stacking_v2의 정확도가 더 높음
# final_estimator(최종 예측기)로 LogisticRegression 선택
# - 랜덤포레스트는 복잡하고 강력한 모델이지만, 스태킹에서는 기반 모델의 예측 결과
#   (작은 feature set)를 다루므로 -> 복잡한 모델보다 단순한 로지스틱 회귀가 
#   더 일반화에 유리한 경우가 많음

# --------------------------------------------------------------
# 3단계 실습
# 스케일링 적용 후 성능 비교
# KNN과 LogisticRegression은 스케일에 민감하므로 스케일링 전/후 비교
# 스케일링: 데이터의 크기를 일정한 범위로 바꿔주는 작업
# --------------------------------------------------------------
# (1) StandardScaler 적용
from sklearn.preprocessing import StandardScaler

# 스케일링 객체 생성
# StandardScaler(): 가장 널리 쓰이는 스케일링 방법
scaler = StandardScaler()   

# 훈련 데이터 기준으로 학습 및 변환
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test)

# (2) 스케일링된 데이터로 stacking_v2 다시 학습
stacking_scaled = StackingClassifier(
    estimators=[('dt', dt), ('lr', lr), ('knn', knn)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
stacking_scaled.fit(X_train_scaled, y_train)
pred_scaled = stacking_scaled.predict(X_val_scaled)
print("스케일링 적용 후 stacking 정확도", accuracy_score(y_val, pred_scaled))

# 스케일링은 항상 해보는 게 좋지만, 모든 조합에서 성능 차이를 보장하진 않음
# 다만, 실기 시험에서 "스케일링 적용 여부 판단"과 실험은 중요한 포인트
# KNN, SVM. 로지스틱 회귀 등은 스케일 영향이 크므로 스케일링 필수!
# 트리 계열 모델(DT, RF, XGBoost 등)은 스케일 무관하므로 굳이 안 해도 됨

# --------------------------------------------------------------
# 🔽 GridSearchCV 튜닝 실습
# --------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

# 랜덤포레스트 파라미터 튜닝
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10]
}

grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)

grid_rf.fit(X_train, y_train)
print("Best Params:", grid_rf.best_params_)
print("Best Score:", grid_rf.best_score_)

# 최종 스태킹 모델 만들기 & 평가
# 최적 파라미터로 듀닝된 랜덤포레스트 모델 사용
best_rf = RandomForestClassifier(
    max_depth=5,
    n_estimators=50,
    random_state=42
)

# 최종 스태킹 모델 정의
stacking_final = StackingClassifier(
    estimators=[('dt', dt), ('lr', lr), ('knn', knn)],
    final_estimator=best_rf,
    cv=5
)

# 학습 & 예측
stacking_final.fit(X_train, y_train)
pred_final = stacking_final.predict(X_val)

# 정확도 확인
print("최종 스태킹 정확도:", accuracy_score(y_val, pred_final))

# 전체 학습 데이터로 다시 학습 후 test 데이터 예측
stacking_final.fit(X, y)
final_pred = stacking_final.predict(test)

# 제출 파일 저장
submission['Survived'] = final_pred
submission['PassengerId'] = test_passenger_ids
submission.to_csv('final_submission.csv', index=False)


# 학습할 때 쓰는 데이터 vs 예측할 때 쓰는 데이터
# 모델이 이미 본 데이터로 예측하면 정확도가 뻥튀기되기 때문에
# 실제 성능을 평가하려면 "처음 보는 데이터"로 테스트

# X_train: 학습에 사용할 입력 특성(features)
# y_train: 학습에 사용할 정답(label, 타깃)
# X_val: 예측할 때 사용할 입력 특성(검증용)
# y_val: 예측값과 비교할 정답(label)(검증용용)

# fit vs predict 매개변수
# (1) 학습할 때는 fit(X_train, y_train)
# 모델이 X_train을 보고 y_train을 예측하는 방법을 학습
# 즉, X_train -> y_train을 맞추는 규칙을 찾아가는 과정이 "fit"

# (2) 예측할 때는 predict(X_val)
# X_val 은 처음 보는 문제들
# 여기에 대해 모델이 정답(y_val)을 모른 채로 예측해봄 -> pred

# (3) 평가할 때는 accuracy_score(y_val, pred)
# 모델이 예측한 값 pred와 실제 정답 y_val을 비교해서 정확도 계산