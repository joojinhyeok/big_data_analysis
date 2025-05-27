# """
# ** 교차검증(Cross Validation) **
# - 교차 검증이 필요한 이유
# - (1) 모델이 과적합 되는 걸 막기 위해
# - (2) 단순히 train_test_split()으로 한 번만 나누는 건 신뢰도가 낮음
# - (3) 다양한 데이터 조합에서 테스트해보면, 모델의 일반화 성능을 더 잘 알 수 있음

# -> 가장 많이 쓰는 방법: K-Fold 교차검증
# """

# 1. 데이터 불러오기 및 전처리
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv("C:/csv/train.csv")
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
train = train.dropna(subset=features)

# PassengerId 따로 저장 (제출 파일 생성용)
passenger_id_all = train['PassengerId']

X = pd.get_dummies(train[features], drop_first=True)
y = train['Survived']

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# PassengerId 포함하여 train_test_split 수행
X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
    X, y, passenger_id_all, test_size=0.2, random_state=42
)

# 2. 교차검증을 위한 모델 정의 및 준비
# ** KFold란? **
# 데이터를 K개로 나눈 뒤, 그중 하나를 테스트셋, 나머지를 훈련셋으로 반복
# ex) cv=5 -> 5등분해서 5번 훈련/평가 반복
model = RandomForestClassifier(random_state=42)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 교차검증 정확도 측정
scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

print("각 Fold의 정확도: ", scores)
print("평균 정확도: ", np.mean(scores))

# """
# 출력 결과 (예시, 실행할 때마다 약간 달라질 수 있음)
# 각 Fold의 정확도:  [0.80701754 0.78947368 0.77192982 0.81578947 0.76106195]
# 평균 정확도:  0.7890544946436888

# --> cv=5이므로 5번 반복하면서, 각각의 검증 정확도를 출력함
# """

# ***** GridSearchCV를 이용한 하이퍼파라미터 튜닝 *****

# 1) 사용할 모델
model = RandomForestClassifier(random_state=42)

# 2) 튜닝할 하이퍼파라미터 조합 만들기
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}

# 3) GridSearchCV 객체 생성 (cv=5는 교차검증 5번)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1)

# 4) 모델 학습 (12개의 하이퍼파라미터 조합 각각 학습됨)
grid.fit(X_train, y_train)

# 5) 결과 확인
print("최적의 파라미터:", grid.best_params_)
print("최고의 정확도 (평균 교차검증):", grid.best_score_)

# 6) 테스트 데이터로 예측
pred = grid.best_estimator_.predict(X_test)

# 7) 제출용 DataFrame 생성
submission = pd.DataFrame({
    'PassengerId': pid_test,
    'Survived': pred
})

# 8) CSV 파일로 저장
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv 저장 완료!")
