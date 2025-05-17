# 📌 라이브러리 불러오기
import pandas as pd  # 데이터 불러오기 및 전처리용
import joblib        # 피처 순서 저장 등 파일 저장/불러오기용
from sklearn.model_selection import train_test_split  # 학습/검증 데이터 분할
from sklearn.metrics import accuracy_score            # 예측 결과 평가 (정확도 계산)

# 머신러닝 모델들 (단일 및 앙상블)
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier  # 고성능 트리 기반 부스팅 모델

# ------------------------------------------------------------------------------

# 1️⃣ 학습용 데이터 로드 및 전처리
train = pd.read_csv('C:/csv/train.csv')  # train.csv 파일 불러오기

# 사용할 피처 정의 및 결측치 제거
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
train = train.dropna(subset=features)

# 범주형 데이터(문자열)를 숫자로 변환 (One-Hot Encoding)
X = pd.get_dummies(train[features], drop_first=True)

# 생존 여부(label) 정의
y = train['Survived']

# 학습용(X_train)과 검증용(X_val) 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------------------

# 2️⃣ 개별 모델 정의
lr = LogisticRegression(max_iter=1000)         # 로지스틱 회귀: 선형 분류 모델
rf = RandomForestClassifier()                  # 랜덤 포레스트: 여러 트리를 평균내는 앙상블
knn = KNeighborsClassifier()                   # KNN: 가장 가까운 데이터들의 클래스를 따름
xgb = XGBClassifier(eval_metric='logloss')     # XGBoost: 성능 좋은 부스팅 모델 (트리 기반)

# ------------------------------------------------------------------------------

# 3️⃣ 투표 기반 앙상블 모델 정의 (VotingClassifier)
# ▶ 여러 모델의 예측을 '투표'로 결정함

# Hard Voting: 각 모델의 최종 예측 결과(클래스)를 다수결로 선택
voting_hard = VotingClassifier(estimators=[
    ('lr', lr), ('rf', rf), ('knn', knn)
], voting='hard')

# Soft Voting: 각 모델의 확률 예측 결과를 평균내어 결정
voting_soft = VotingClassifier(estimators=[
    ('lr', lr), ('rf', rf), ('xgb', xgb)
], voting='soft')

# ------------------------------------------------------------------------------

# 4️⃣ 스태킹 기반 앙상블 모델 정의 (StackingClassifier)
# ▶ 여러 모델의 예측을 하나의 '최종 모델'이 다시 학습하여 결합함

stacking = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],              # 1층(base) 모델
    final_estimator=LogisticRegression()               # 2층(meta) 모델
)

# ------------------------------------------------------------------------------

# 5️⃣ 모델 평가 함수 정의
def evaluate_model(name, model):
    model.fit(X_train, y_train)                         # 모델 학습
    preds = model.predict(X_val)                        # 검증 데이터로 예측 수행
    acc = accuracy_score(y_val, preds)                  # 예측 정확도 계산
    print(f"{name} Accuracy: {acc:.4f}")                # 결과 출력

# ------------------------------------------------------------------------------

# 6️⃣ 각 앙상블 모델 성능 평가
# -> Voting과 Stacking 모델들이 검증 데이터셋에서 얼마나 정확한 예측을 하는지 확인
evaluate_model("Voting (Hard)", voting_hard)
evaluate_model("Voting (Soft)", voting_soft)
evaluate_model("Stacking", stacking)
