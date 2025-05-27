# Day 11 - VotingClassifier 앙상블 모델 만들기 (Soft Voting 버전)

# ----------------------------------------------------------
# 1단계 : 기본 VotingClassifier 실습
# ----------------------------------------------------------

# (1) 필요한 라이브러리 불러오기
import pandas as pd  # 데이터프레임 형태로 데이터를 다루기 위한 라이브러리
from sklearn.model_selection import train_test_split  # 학습/테스트 데이터 분할 함수
from sklearn.ensemble import VotingClassifier  # 여러 모델을 조합하는 앙상블 분류기
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델
from sklearn.tree import DecisionTreeClassifier  # 결정 트리 분류기
from sklearn.neighbors import KNeighborsClassifier  # K-최근접 이웃 분류기
from sklearn.metrics import accuracy_score  # 모델 성능 평가 지표 중 하나인 정확도

# (2) 데이터 불러오기 및 전처리
train = pd.read_csv('C:/csv/train.csv')  # Titanic 생존자 예측 데이터셋 로드

# (3) 결측치 처리 (누락된 값 채우기)
train['Age'] = train['Age'].fillna(train['Age'].median())  # Age 열의 결측치는 중앙값으로 채움
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])  # Embarked는 최빈값으로 채움

# (4) 범주형 -> 숫자형 변환 (모델은 숫자만 인식 가능하기 때문)
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})  # 성별: 남자→0, 여자→1
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})  # 탑승항: S→0, C→1, Q→2

# (5) feature, target 설정
# 모델에 넣을 입력 데이터(features) 선택
X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
# 예측할 대상값(target): 생존 여부
y = train['Survived']

# 학습용 데이터와 테스트용 데이터로 분리 (80%:20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# (6) 개별 모델 정의
# 각각 다른 특성을 가진 분류 모델을 정의
Logistic_model = LogisticRegression(max_iter=1000)  # 선형 모델, 빠르고 해석 용이
Tree_model = DecisionTreeClassifier()  # 규칙 기반의 트리 구조 모델, 직관적
KN_model = KNeighborsClassifier()  # 주변 이웃 데이터를 보고 판단하는 비모수 모델

# (7) Soft VotingClassifier 구성
# 여러 모델의 예측 확률을 평균 내서 가장 높은 확률을 가진 클래스를 최종 예측으로 선택
voting_clf_soft = VotingClassifier(
    estimators=[
        ('lr', Logistic_model),  # 이름 'lr'으로 로지스틱 회귀 모델 추가
        ('T', Tree_model),       # 이름 'T'으로 결정 트리 모델 추가
        ('Kn', KN_model)         # 이름 'Kn'으로 KNN 모델 추가
    ],
    voting='soft'  # soft voting: 예측 확률을 평균해서 예측
    # → 예: 모델1은 0.6, 모델2는 0.8, 모델3은 0.4 확률로 1이라 예측했다면,
    # 평균 0.6이므로 최종 예측은 1
)

# 학습 데이터를 이용해 Soft Voting 모델 학습
voting_clf_soft.fit(X_train, y_train)

# 테스트 데이터로 예측 수행
y_pred = voting_clf_soft.predict(X_test)

# 예측 결과와 실제 생존값을 비교해 정확도 계산
acc_soft = accuracy_score(y_test, y_pred)
print(f'Soft Voting Accuracy: {acc_soft:.4f}')  # 예: Soft Voting Accuracy: 0.8212
