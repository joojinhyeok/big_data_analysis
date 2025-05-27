# --------------------------------------------------------------------
# 1. StackingClassifier 실습
# --------------------------------------------------------------------

# 1. 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# 2. 데이터 불러오기
df = pd.read_csv('C:/csv/train.csv')  # Titanic 데이터 예시

# 3. 전처리 (간단 버전)
df = df[['Pclass', 'Sex', 'Age', 'Survived']].dropna()
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # male=1, female=0

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 구성
estimators = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('knn', KNeighborsClassifier())
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

# 5. 학습 및 예측
stack_model.fit(X_train, y_train)
pred = stack_model.predict(X_test)

# 6. 평가
print("StackingClassifier Accuracy:", accuracy_score(y_test, pred))

# --------------------------------------------------------------------
# 2. Stacking 모델 성능 향상 - GridSearchCV 적용
# --------------------------------------------------------------------
# 1. 라이브러리 불러오기
from sklearn.model_selection import GridSearchCV

# 2. 기반 모델 중 하나인 DecisionTree에 대해 GridSearchCV 적용
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6]
}

# 3. 기본 모델 정의
dt = DecisionTreeClassifier(random_state=42)

# 4. 그리드 서치 실행
grid_dt = GridSearchCV(dt, param_grid, cv=5, n_jobs=1)
grid_dt.fit(X_train, y_train)

print("Best Params (DecisionTree): ", grid_dt.best_params_)

# 5. 튜닝된 모델을 기반 모델로 사용하여 스태킹 구성
estimators = [
    ('dt', grid_dt.best_estimator_), # 튜닝된 결정트리 사용
    ('knn', KNeighborsClassifier())
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

# 6. 학습 및 평가
stack_model.fit(X_train, y_train)
pred = stack_model.predict(X_test)
print("Tuned StackingClassifier Accuracy: ", accuracy_score(y_test, pred))

# --------------------------------------------------------------------
# 3. 교차검증으로 평가해보기
# - 정확도 비교 시 단일 테스트셋 말고 K-Fold 교차검증으로 "평균 정확도" 확인
#   하는게 더 안정적인 평가
# --------------------------------------------------------------------
# 1. 라이브러리 가져오기
from sklearn.model_selection import cross_val_score

# 2. 교차검증으로 평균 정확도 확인
scores = cross_val_score(stack_model, X, y, cv=5, scoring='accuracy')

print("교차검증 정확도 각(Fold)", scores)
print("평균 정확도: ", scores.mean())

# --------------------------------------------------------------------
# 4. RandomizedSearchCV 적용 실습
# --------------------------------------------------------------------
# 1. 라이브러리 불러오기
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# 2. 탐색 범위 정의 (난수 분포로 정의)
param_dist = {
    'max_depth': randint(3, 10),             # 3~9
    'min_samples_split': randint(2, 10)      # 2~9
}

# 3. 랜덤 서치 실행
random_dt = RandomizedSearchCV(
    estimator = DecisionTreeClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,              # 랜덤하게 10조합 뽑아 시험
    cv=5,
    random_state=42,
    n_jobs=1                # 시험 환경 고려해 1로 고정
)

random_dt.fit(X_train, y_train)

print("Best Parms (Randomized):", random_dt.best_params_)

# 4. 최적 모델을 기반으로 스태킹 구성
estimators = [
    ('dt', random_dt.best_estimator_),  # 튜닝된 결정트리
    ('knn', KNeighborsClassifier())
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

# 5. 학습 및 예측
stack_model.fit(X_train, y_train)
pred = stack_model.predict(X_test)

print("Tuned StackingClassifier Accuracy (Randomized):", accuracy_score(y_test, pred))

### ✔ 모델 정확도 비교

# - 기본 스태킹 모델 정확도: **0.748**
# - GridSearchCV 튜닝 결과: **0.741**
# - RandomizedSearchCV 튜닝 결과: **0.734**
# - 교차검증 평균 정확도: **0.798**

# 🧠 **분석**: 튜닝을 통해 최적의 파라미터를 찾았지만, 실제 테스트셋에서는 정확도가 다소 낮게 나올 수 있다.  
# 이는 교차검증과 실제 분할의 차이, 데이터 적합도, 튜닝 범위 제한 등의 이유로 분석된다.
