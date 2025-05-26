# ----------------------------------------------------------
# BaggingClassifier - 같은 모델 여러 개 + 샘플을 무작위로 뽑아 훈련
# ----------------------------------------------------------

# (1) 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# (2) 데이터 불러오기 및 전처리
train = pd.read_csv('C:/csv/train.csv')

# (3) 결측치 처리
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

# (4) 범주형 -> 숫자형 변환
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'S':0, 'C':1, 'Q':2})

# (5) feature, target 설정
X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (2) 기본 결정트리로 Bagging 구성
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,   # 트리 개수
    random_state=42,
    n_jobs=1           # cpu 병렬처리x
)

bagging_clf.fit(X_train, y_train)
y_pred_bag = bagging_clf.predict(X_test)

acc_bag = accuracy_score(y_test, y_pred_bag)
print(f'BaggingClassifier Accuracy: {acc_bag:.4f}')


# ----------------------------------------------------------
# RandomForestClassifier - Bagging + 피처도 랜덤으로 뽑음
#                           (그래서 더 다양성 높음)
# ----------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
print(f'RandomForestClassifier Accuracy: {acc_rf:.4f}')


# 비교 정리
print(f'✅ Bagging 정확도: {acc_bag:.4f}')
print(f'🌲 RandomForest 정확도: {acc_rf:.4f}')

# ----------------------------------------------------------
# StackingClassifier - 여러 모델 조합 + 최종 메타모델로 예측
# ----------------------------------------------------------
from sklearn.ensemble import StackingClassifier

# Base 모델 3개 정의
base_models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier()),
    ('knn', KNeighborsClassifier())
]

# 최종 예측을 담당할 메타 모델
final_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 스태킹 모델 구성
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=final_model,
    cv=5,
    n_jobs=1
)

# 학습
stacking_clf.fit(X_train, y_train)

# 예측
y_pred_stack = stacking_clf.predict(X_test)

# 정확도 평가
acc_stack = accuracy_score(y_test, y_pred_stack)
print(f'StackingClassifier Accuracy: {acc_stack:.4f}')

# 전체 비교 출력
print("\n📊 모델 성능 비교")
print(f'✅ Bagging 정확도: {acc_bag:.4f}')
print(f'🌲 RandomForest 정확도: {acc_rf:.4f}')
print(f'🔀 Stacking 정확도: {acc_stack:.4f}')
