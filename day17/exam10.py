# 10. apply() 함수 + 사용자 정의 함수
# Titanic 데이터셋의 Name 컬럼에는 승객의 이름과 함께 Mr. Mrs. Miss. 등의
# title이 포함돼 있다.
# Name 컬럼에서 title만 추출해 새로운 컬럼 Title을 생성하시오.
# 생성 후 Title 컬럼의 고유값 목록을 출력하시오

import pandas as pd

train = pd.read_csv('C:/csv/train.csv')

# 함수 정의
"""
ex) "Brand, MR. Owen Harris"
    1. name.split(',')
        -> ['Brand', ' Mr. Owen Harris']
    2. [1].split('.')
        -> [' Mr', ' Owen Harris']
    3. [0].strip()
        -> 'Mr'
"""

def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()

# apply로 전체 Name 컬럼에 적용
train['Title'] = train['Name'].apply(extract_title)

# 고유값 출력
print(train['Title'].unique())