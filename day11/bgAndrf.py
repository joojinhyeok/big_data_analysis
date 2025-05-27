# ----------------------------------------------------------
# BaggingClassifier - 같은 모델 여러 개 + 샘플을 무작위로 뽑아 훈련
# ----------------------------------------------------------

# (1) 라이브러리 불러오기
import pandas as pd  # 데이터 분석용 라이브러리
from sklearn.model_selection import train_test_split  # 학습/테스트 분리 함수
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델
from sklearn.tree import DecisionTreeClassifier  # 결정 트리 모델
from sklearn.neighbors import KNeighborsClassifier  # KNN 모델
from sklearn.metrics import accuracy_score  # 모델 성능 평가 (정확도)
from sklearn.ensemble import BaggingClassifier  # Bagging 앙상블 모델

# (2) 데이터 불러오기 및 전처리
train = pd.read_csv('C:/csv/train.csv')  # Titanic 생존자 예측 데이터셋 로드

# (3) 결측치 처리 (누락된 데이터 채우기)
train['Age'] = train['Age'].fillna(train['Age'].median())  # 나이: 중앙값으로 대체
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])  # 탑승항: 최빈값으로 대체

# (4) 범주형 -> 숫자형 변환
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})  # 남:0, 여:1
train['Embarked'] = train['Embarked'].map({'S':0, 'C':1, 'Q':2})  # S:0, C:1, Q:2

# (5) feature, target 설정
X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]  # 입력 변수들
y = train['Survived']  # 예측 대상 (0:사망, 1:생존)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (6) Bagging 모델 구성 - 같은 결정트리를 여러 개 복사해서 사용 (샘플은 무작위)
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),  # 기본 모델은 결정트리
    n_estimators=100,   # 트리 100개 사용
    random_state=42,    # 랜덤성 고정 (재현 가능성)
    n_jobs=1            # CPU 병렬 처리 안 함 (-1로 하면 모든 코어 사용)
)

# 모델 학습
bagging_clf.fit(X_train, y_train)

# 예측 및 정확도 평가
y_pred_bag = bagging_clf.predict(X_test)
acc_bag = accuracy_score(y_test, y_pred_bag)
print(f'BaggingClassifier Accuracy: {acc_bag:.4f}')


# ----------------------------------------------------------
# RandomForestClassifier - Bagging + 피처도 랜덤으로 뽑음
# ----------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier  # 랜덤포레스트 불러오기

# 랜덤포레스트는 결정트리를 여러 개 만들되, 피처도 무작위로 뽑아서 다양성 높임
rf_clf = RandomForestClassifier(
    n_estimators=100,    # 트리 개수
    max_depth=None,      # 트리 깊이는 제한 없음
    random_state=42,     # 랜덤 고정
    n_jobs=-1            # 모든 CPU 코어 사용해서 병렬 처리
)

# 학습 및 예측
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# 정확도 평가
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f'RandomForestClassifier Accuracy: {acc_rf:.4f}')


# 비교 정리 출력
print(f'Bagging Accuracy: {acc_bag:.4f}')
print(f'RandomForest Accuracy: {acc_rf:.4f}')


# ----------------------------------------------------------
# StackingClassifier - 여러 모델 조합 + 최종 메타모델로 예측
# ----------------------------------------------------------
from sklearn.ensemble import StackingClassifier  # 스태킹 앙상블 모델

# Base 모델 3개 정의 (각각 다른 유형의 모델을 조합)
base_models = [
    ('lr', LogisticRegression(max_iter=1000)),  # 로지스틱 회귀
    ('dt', DecisionTreeClassifier()),           # 결정트리
    ('knn', KNeighborsClassifier())             # KNN
]

# 최종 예측을 담당할 메타 모델 설정 (보통 성능 좋은 모델 선택)
final_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 스태킹 모델 구성
stacking_clf = StackingClassifier(
    estimators=base_models,         # 기본 모델 리스트
    final_estimator=final_model,   # 최종 예측 담당 메타 모델
    cv=5,                           # 교차검증 folds 수 (과적합 방지)
    n_jobs=1                        # 병렬처리 없음
)

# 학습
stacking_clf.fit(X_train, y_train)

# 예측 및 평가
y_pred_stack = stacking_clf.predict(X_test)
acc_stack = accuracy_score(y_test, y_pred_stack)
print(f'StackingClassifier Accuracy: {acc_stack:.4f}')


# 전체 모델 성능 비교 출력
print()
print(f'Bagging Accuracy: {acc_bag:.4f}')
print(f'RandomForest Accuracy: {acc_rf:.4f}')
print(f'Stacking Accuracy: {acc_stack:.4f}')
