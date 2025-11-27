import pandas as pd

train = pd.read_csv('csv/train.csv')

# [문제 1] 이상치 대체 (Winsorization)
# Fare(요금) 데이터를 내림차순으로 정렬했을 때, 상위 10개(1등~10등)의 데이터를 11번째(11등) 데이터의 값으로 변경하시오. 변경 후, Fare 컬럼의 평균을 구하시오. 
# (단, 소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력)
train = train.sort_values('Fare', ascending=False)
# print(train)

# 상위 10개 데이터 덮어쓰기 
train['Fare'].iloc[:10] = train['Fare'].iloc[10]
# print(train)

Ans1 = train['Fare'].mean()

# print(round(Ans1, 2)) # 답: 31.21

# ========================================================================================================================================

# [문제 2] 파생변수 & 조건부 집계
# SibSp(형제자매)와 Parch(부모자녀)를 합치고 본인(1)을 더해 FamilySize 컬럼을 만드시오. FamilySize가 5명 이상(>= 5)인 대가족 그룹의 생존율(Survived의 평균)을 구하시오. 
# (정답은 소수점 넷째 자리에서 반올림하여 셋째 자리까지 출력)
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

Ans2 = train[train['FamilySize'] >= 5]['Survived'].mean()

# print(round(Ans2, 3)) # 답: 0.107 -> 1번문제에서 1~10까지 데이터를 11번째로 덮어쓴 기준
                      # 답: 0.161 -> 기존 데이터로 구한 기준
# ========================================================================================================================================

# [문제 3] 문자열 분리 (단어 개수)
# Name 컬럼에서 공백을 기준으로 단어를 나누었을 때, 이름에 포함된 단어의 개수가 4개인 승객은 몇 명인가? 
# (예: "Braund, Mr. Owen Harris" -> 단어 4개) (정답은 정수로 출력)

train['SplitName'] = train['Name'].str.split(" ")

# print(train['SplitName'])

Ans3 = train[train['SplitName'].str.len() == 4]

print(len(Ans3)) # 답: 388 -> 기존 데이터로 구한 기준
                 # 답: 393 -> 1번 문제에서 1~10까지 데이터를 11번째로 덮어쓴 기준
