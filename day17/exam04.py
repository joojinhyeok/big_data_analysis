# 4. 범주형 변수 처리
# Titanic 데이터셋에서 Sex(성별) 컬럼을 0과 1의 숫자로 변환하시오.
# male은 0, female은 1로 변환할 것
# 변환 후 Sex 컬럼의 고유값(unique)을 출력하시오

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')

# replace() 활용해서 male과 female값을 0과 1로 변경
train['Sex'] = train['Sex'].replace({'male': 0, 'female' : 1})

# unique() 활용해서 고유값 출력
print(train['Sex'].unique())