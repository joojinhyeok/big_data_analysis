# 📌 필요한 라이브러리 불러오기
import pandas as pd  # 데이터 불러오기 및 전처리용
import joblib        # 학습에 사용한 피처 정보 불러오기용
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier  # (이번 실습에선 사용 안 했지만 여유롭게 불러옴)

# ------------------------------------------------------------------------------

# 1️⃣ test 데이터 불러오기
test = pd.read_csv('C:/csv/test.csv')  # 실제 대회에서 제출용으로 주어지는 테스트 데이터셋

# ------------------------------------------------------------------------------

# 2️⃣ 피처 구성 동일하게 하기 위한 전처리
# 학습(train) 때 사용했던 피처들과 일치하도록 전처리
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
test = test.dropna(subset=features)                  # 결측치 제거
test_passenger_id = test['PassengerId']              # 예측값과 함께 저장할 ID
X_test = pd.get_dummies(test[features], drop_first=True)  # One-Hot 인코딩 (drop_first=True: 첫 범주는 제거)

# ------------------------------------------------------------------------------

# 3️⃣ 학습에 사용한 피처 목록 불러오기 (순서 및 구성 일치 위해)
model_columns = joblib.load('./model_columns.pkl')  # train 시 저장했던 피처 컬럼 순서 정보

# ------------------------------------------------------------------------------

# 4️⃣ test셋 컬럼 정리 (누락된 피처 채우기 & 순서 맞추기)
for col in model_columns:
    if col not in X_test.columns:
        X_test[col] = 0  # 누락된 피처는 값 0으로 채움 (ex. 해당 범주가 test에 없을 경우)
X_test = X_test[model_columns]  # 컬럼 순서 일치

# ------------------------------------------------------------------------------

# 5️⃣ 최종 모델 정의 – VotingClassifier (Hard Voting)
# 3개의 모델의 예측 결과를 다수결로 결정함
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
knn = KNeighborsClassifier()

voting_hard = VotingClassifier(estimators=[
    ('lr', lr), ('rf', rf), ('knn', knn)
], voting='hard')

# ------------------------------------------------------------------------------

# 6️⃣ 전체 train 데이터를 불러와 최종 모델 재학습
train = pd.read_csv('C:/csv/train.csv')
train = train.dropna(subset=features)
X = pd.get_dummies(train[features], drop_first=True)
y = train['Survived']

# 누락 피처 채우기 & 컬럼 순서 맞추기
for col in model_columns:
    if col not in X.columns:
        X[col] = 0
X = X[model_columns]

# 최종 모델 학습
voting_hard.fit(X, y)

# ------------------------------------------------------------------------------

# 7️⃣ test 데이터 예측 및 제출 파일 생성
preds = voting_hard.predict(X_test)  # 예측 수행

# 제출용 데이터프레임 생성 (ID + 예측 결과)
submission = pd.DataFrame({
    'PassengerId': test_passenger_id,
    'Survived': preds
})

# CSV 파일로 저장 (index=False: 인덱스는 제외)
submission.to_csv('submission.csv', index=False)

# 결과 메시지 출력
print("✅ submission.csv 생성 완료!")
