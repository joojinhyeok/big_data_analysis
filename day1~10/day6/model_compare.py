# 📌 필요한 라이브러리 불러오기
import pandas as pd  # 데이터프레임 형태로 데이터를 다루기 위한 라이브러리
import joblib        # 학습된 모델이나 데이터를 파일로 저장하고 불러오기 위한 라이브러리
import numpy as np   # 숫자 계산 및 통계 처리용 라이브러리

# 평가 및 모델 관련 함수 불러오기
from sklearn.metrics import accuracy_score               # 예측 결과의 정확도를 계산하는 함수
from sklearn.linear_model import LogisticRegression      # 로지스틱 회귀 모델 (선형 분류)
from sklearn.ensemble import RandomForestClassifier      # 랜덤포레스트 (트리 기반 앙상블 모델)
from sklearn.neighbors import KNeighborsClassifier       # K-최근접 이웃(KNN) 분류기
from sklearn.svm import SVC                              # 서포트 벡터 머신(SVM)
from xgboost import XGBClassifier                        # XGBoost 모델 (트리 기반 부스팅)
from sklearn.model_selection import cross_val_score      # 교차검증 수행 함수

# ------------------------------------------------------------------------------

# 1️⃣ 학습/검증 데이터 로드 (5일차에서 저장한 데이터 불러오기)
# split_data.pkl에는 X_train, X_val, y_train, y_val가 저장되어 있음
X_train, X_val, y_train, y_val = joblib.load('./split_data.pkl')

# ------------------------------------------------------------------------------

# 2️⃣ 전체 train 데이터를 교차검증용으로 다시 불러옴
train = pd.read_csv('C:/csv/train.csv')  # 원본 train.csv 불러오기

# 사용할 입력 피처 정의 (결측치 있는 행은 제거)
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
train = train.dropna(subset=features)

# 문자열 데이터를 숫자로 변환 (One-Hot Encoding)
X = pd.get_dummies(train[features], drop_first=True)

# 정답(label) 데이터 정의
y = train['Survived']

# ------------------------------------------------------------------------------

# 3️⃣ 사용할 머신러닝 분류 모델들을 딕셔너리 형태로 정의
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),  # 선형 모델 (확률 기반)
    "Random Forest": RandomForestClassifier(),                 # 트리 앙상블 모델
    "KNN": KNeighborsClassifier(),                             # 가장 가까운 이웃 기반 모델
    "SVM": SVC(),                                              # 벡터 기반 경계 분류 모델
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")  # 부스팅 기반 고성능 모델
}

# ------------------------------------------------------------------------------

# 4️⃣ 검증용 데이터셋(X_val)에 대해 각 모델의 정확도 출력
print("📌 검증셋 기반 정확도 (Validation Accuracy)")
for name, model in models.items():
    model.fit(X_train, y_train)              # 모델 학습
    preds = model.predict(X_val)             # 검증셋 예측
    acc = accuracy_score(y_val, preds)       # 정확도 계산
    print(f"{name} Accuracy: {acc:.4f}")     # 결과 출력

# ------------------------------------------------------------------------------

# 5️⃣ 교차검증(Cross Validation)을 통해 평균 정확도 및 모델 안정성 평가
print("\n📌 교차검증 (5-Fold) 기반 평균 정확도")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)  # 데이터를 5개로 나눠 평균 정확도 측정
    print(f"{name} 평균 정확도: {np.mean(scores):.4f}, 표준편차: {np.std(scores):.4f}")
