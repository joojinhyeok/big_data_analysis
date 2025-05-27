# Day 11 - VotingClassifier 앙상블 모델 만들기

# ----------------------------------------------------------
# 1단계 : 기본 VotingClassifier 실습
# ----------------------------------------------------------

# (1) 필요한 라이브러리 불러오기
import pandas as pd  # 데이터 분석을 위한 라이브러리
from sklearn.model_selection import train_test_split  # 데이터 분할 (학습용 vs 테스트용)
from sklearn.ensemble import VotingClassifier  # 앙상블 모델 (여러 모델의 예측을 모아서 결정)
from sklearn.linear_model import LogisticRegression  # 선형 회귀 기반 분류 모델
from sklearn.tree import DecisionTreeClassifier  # 트리 기반 분류 모델
from sklearn.neighbors import KNeighborsClassifier  # 가까운 이웃 기반 분류 모델
from sklearn.metrics import accuracy_score  # 정확도 평가 함수

# (2) 데이터 불러오기 및 전처리
train = pd.read_csv('C:/csv/train.csv')  # Titanic 생존자 데이터셋 불러오기

# (3) 결측치 처리
train['Age'] = train['Age'].fillna(train['Age'].median())  # 나이(Age) 결측치는 중앙값으로 채움
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])  # 탑승항(Embarked) 결측치는 최빈값으로 채움

# (4) 범주형 -> 숫자형 변환 (모델이 숫자만 이해할 수 있기 때문)
"""
get_dummies() 대신 map()을 쓴 이유
- 순서가 없고, 단순하게 0, 1, 2와 같은 숫자로 치환만 하면 되는
- 이진/소수 클래스이기 때문
"""
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})  # 남성:0, 여성:1
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})  # S:0, C:1, Q:2로 변환

# (5) feature, target 설정
X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]  # 입력값(특징들) - features
y = train['Survived']  # 예측할 값(생존 여부) - target

# 훈련 데이터와 테스트 데이터를 8:2 비율로 분할 (랜덤시드는 42로 고정)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (6) 개별 모델 정의
Logistic_model = LogisticRegression(max_iter=1000)  # 로지스틱 회귀: 선형적인 분류모델
Tree_model = DecisionTreeClassifier()  # 결정 트리 모델: 조건에 따라 분기해서 예측
KN_model = KNeighborsClassifier()  # K-최근접 이웃: 주변 가까운 데이터 기반 예측

# (7) Hard VotingClassifier 구성
# 각 모델의 예측 결과 중 가장 많이 선택된 클래스를 최종 예측으로 사용함 (다수결 방식)
voting_clf = VotingClassifier(estimators=[
    ('lr', Logistic_model),  # 첫 번째 모델: 로지스틱 회귀
    ('T', Tree_model),       # 두 번째 모델: 결정 트리
    ('Kn', KN_model)         # 세 번째 모델: KNN
], voting='hard')  # 'hard': 다수결 투표 방식, 'soft': 확률 평균 방식

# 학습 데이터로 모델 학습
voting_clf.fit(X_train, y_train)

# 테스트 데이터로 예측 수행
y_pred = voting_clf.predict(X_test)

# 정확도 출력 (예측값과 실제값 비교)
acc = accuracy_score(y_test, y_pred)
print(f'Hard Voting Accuracy: {acc:.4f}')
