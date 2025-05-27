# 1. 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint

# 2. 데이터 불러오기
df = pd.read_csv('C:/csv/train.csv')  # Titanic 데이터 예시

# 3. 전처리
df = df[['Pclass', 'Sex', 'Age', 'Survived']].dropna()
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

X = df.drop('Survived', axis=1)
y = df['Survived']

# 4. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. RandomizedSearchCV로 DecisionTree 튜닝
param_dist = {
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 10)
}

random_dt = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=1
)

random_dt.fit(X_train, y_train)
best_dt = random_dt.best_estimator_
print("Best Params:", random_dt.best_params_)

# 6. 스태킹 모델 구성
estimators = [
    ('dt', best_dt),
    ('knn', KNeighborsClassifier())
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

# 7. 학습 및 예측
stack_model.fit(X_train, y_train)
pred = stack_model.predict(X_test)

# 8. 정확도 평가
print("최종 정확도:", accuracy_score(y_test, pred))

# 9. 교차검증
cv_scores = cross_val_score(stack_model, X, y, cv=5)
print("교차검증 평균 정확도:", cv_scores.mean())
