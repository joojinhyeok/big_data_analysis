# 1단계 : 데이터 불러오기 및 탐색
import pandas as pd

train = pd.read_csv('C:/csv/train.csv') # 모델 훈련할 때 사용
test = pd.read_csv('C:/csv/test.csv')   # 예측만 할 때 사용

print(train.head()) # 상위 5개 데이터 출력
print(train.info()) # 데이터 정보 출력

# ---------------------------------------------------
# 2단계 : 결측치 처리 + 범주형 변수 인코딩
# train.info()로 결측치 확인한 결과
# Age: 714/819 약 20% 결측   -> 결측 채워서 사용
# Cabin: 204/891 약 77% 결측 -> 결측률 너무 높으니 삭제
# Embarked: 889/891 2개 결측 -> 결측 채워서 사용

# 결측치 처리(fillna(), drop())
# 불필요한 문자형 컬럼 제거(Name, Ticket, Cabin)
# 범주형 변수 인코딩 (get_dummies())
# 훈련/테스트 데이터셋 준비(X_train, y_train, X_test)
# ---------------------------------------------------
# (1) Age 결측치는 평균으로 채움 -> test 데이터에도 결측치가 있을 수 있기 때문에
train['Age'].fillna(train['Age'].mean())
test['Age'].fillna(test['Age'].mean())

# (2) Embarked 결측치는 최빈값으로 채움
train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked'].fillna(test['Embarked'].mode()[0])

# (3) Fare도 test에서 결측 있을 수 있음 -> 평균으로 채움
test['Fare'].fillna(test['Fare'].mean())

# (4) Cabin은 삭제(결측치 너무 많음)
train.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True, errors='ignore')
train.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True, errors='ignore')

# (5) 범주형 변수를 수치형으로 바꿈(One-hot encoding)
train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)

# (6) Feature/Target 분리
# 훈련 데이터
X_train = train.drop(columns=['Survived', 'PassengerId'])
y_train = train['Survived']

# 테스트 데이터 ID 따로 저장
pid_test = test['PassengerId']
X_test = test.drop(columns=['PassengerId'])

# ---------------------------------------------------
# 3단계 : 훈련/검증 데이터 분리(Hold-out 방식)

# 매개변수 설명
# test_size = 0.2 : 검증용 데이터로 20% 사용(나머지 80%는 훈련에 사용)
# random_state = 42 : 데이터 분리 방식 고정(재현 가능성 확보)
# stratify=y_train : 클래스 비율 유지해서 나누기(0과 1의 비율을 유지 하도록 분할)

# 이 과정을 하는 이유
# 전체 데이터로만 모델을 학습하고 바로 테스트셋에 예측하면
# "내가 만든 모델이 진짜 잘 학습된 건가?" 확인할 기회 x
# 그래서 훈련 데이터 안에서 일부를 "검증용"으로 떼어내서 먼저 확인
# ---------------------------------------------------
from sklearn.model_selection import train_test_split

X_train_split, X_valid, y_train_split, y_valid = train_test_split(
    X_train, y_train,
    test_size = 0.2,
    random_state = 42,
    stratify=y_train
) 

# ---------------------------------------------------
# 4단계 : 모델 정의 및 학습
# RandomForestClassifier()는 여러 개의 결정 트리를 앙상블하여
# 예측 정확도를 높이는 모델이며, 기본 설정으로 먼저 학습 후,
# 이후 하이퍼파라미터 튜닝을 통해 성능을 향상 시킬 수 있다.
# ---------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100, # 트리 개수
    max_depth=6,      # 트리 최대 깊이
    random_state=42
)

model.fit(X_train_split, y_train_split)

# ---------------------------------------------------
# 5단계 : GridSearchCV로 최적 하이퍼파라미터 찾기
# GridSearchCV는 하이퍼파라미터 후보 조합을 교차검증으로 평가해
# 최적의 조합을 찾아주는 기법이며, 정확도를 기준으로 가장 성능 좋은
# 모델을 자동으로 선택할 수 있다.
# ---------------------------------------------------
# 1. 튜닝할 파라미터 후보 정의
from sklearn.model_selection import GridSearchCV

param_grid = {
    # 2 x 3 x 2 = 12가지 조합
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}

# 2. GridSearchCV 객체 생성 및 학습
grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,                   # 5-Fold 교차검증
    scoring='accuracy',     # 정확도로 평가
    n_jobs=1                # 병렬처리 x
    # 내부적으로 각 조합마다 5-Fold 교차검증 -> 총 60번 모델 학습 이뤄짐 
)

grid.fit(X_train_split, y_train_split)

# 3. 최적의 파라미터와 정확도 확인
print("최적 하이퍼파라미터: ", grid.best_params_)
print("최고 평균 정확도: ", grid.best_score_)

# 훈련할 때 사용한 컬럼 목록 기준으로 맞춰주기
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# ---------------------------------------------------
# 6단계 : 예측 및 결과 저장
# ---------------------------------------------------
# 1. 최적 모델로 예측 수행
pred = grid.best_estimator_.predict(X_test)

# 2. submission 파일 생성
import pandas as pd
submission_05_24 = pd.DataFrame({
    'PassengerId': pid_test,
    'Survived': pred    # 모델이 예측한 생존 여부
})

submission_05_24.to_csv('submission_05_24.csv', index=False)