# 5. 파생변수 생성
# Titanic 데이터셋에서 "Age(나이)"가 15세 이하인 승객은 "child",
# 16세 이상은 "Adult"로 분류한 새로운 파생변수 AgeGroup 컬럼을 생성하시오.
# 생성 후, AgeGroup 컬럼의 고유값 목록을 출력하시오.

import pandas as pd
import numpy as np

train = pd.read_csv('C:/csv/train.csv')

# Age 기준으로 'child' 또는 'Adult' 값 넣기
# np.where()는 Numpy 버전의 if문이라고 생각. if-else의 형태와 유사
# np.where(조건, 조건이 참일 때 값, 조건이 거짓일 때 값)
train['Agegroup'] = np.where(train['Age'] <= 15, 'child', 'Adult')

# 고유값 출력
print(train['Agegroup'].unique())