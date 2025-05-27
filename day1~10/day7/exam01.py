import pandas as pd

"""
1. 데이터 파일 불러오고 확인하기
"""

# CSV 파일 불러오기
train = pd.read_csv('C:/csv/train.csv')
test = pd.read_csv('C:/csv/test.csv')

# 데이터셋 기본 정보 확인
print("✅ train shape:", train.shape)
print("✅ test shape:", test.shape)

# 컬럼 및 일부 데이터 확인
print("\n📂 train columns:")
print(train.columns)

print("\n🔍 train preview:")
print(train.head())

print("\nℹ️ train info:")
print(train.info())


"""
2. 전처리
"""
# 1. 불필요한 컬럼 제거
drop_cols = ['Name', 'Ticket', 'Cabin'] # -> 제거할 컬럼
train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

# 2. 결측치 처리
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())  # test에만 결측 있음

# 3. 원-핫 인코딩 (drop_first로 더미 변수 다중공선성 제거)
train = pd.get_dummies(train, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# 4. 타깃 분리
y = train['Survived']
X = train.drop(columns=['Survived'])

"""
3. 모델 학습 및 평가
- 여러 모델 중 하나를 선택해 학습
- 훈련/검증 데이터 분리(train_test_split)
- 정확도 평가(accuarcy_score)
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 훈련/검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 모델 정의 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. 검증 데이터로 예측
y_pred = model.predict(X_val)

# 4. 정확도 출력
acc = accuracy_score(y_val, y_pred)
print(f"** 검증 정확도: {acc:.4f}")

"""
4. 모델 저장 & 테스트 데이터 예측
- 학습한 모델과 컬럼 순서 정보를 저장
- 테스트 데이터에 대한 예측 수행
- submission.csv 파일 생성
"""
import joblib

# 1. 모델 저장
joblib.dump(model, 'best_model.pkl')

# 2. 컬럼 순서 저장(테스트셋과 일치시킬 때 필요)
joblib.dump(X.columns, 'model_columns.pkl')

# 3. 테스트 데이터셋 컬럼 맞추기
X_test = test.reindex(columns=X.columns, fill_value=0)

# 4. 예측
predictions = model.predict(X_test)

# 5. 제출 파일 생성
submission = pd.DataFrame({
    'PassengerID': test['PassengerId'],
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)
print("** submission.csv 생성 완료!")