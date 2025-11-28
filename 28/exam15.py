# [제1유형] 데이터 전처리 (3문제)
import pandas as pd

train = pd.read_csv('edu_train.csv')

# Q1. 그룹별 순위 & 평균
# Course(강의)별로 Score(점수)가 가장 높은 상위 5명을 선정하시오. 
# 선정된 우등생들의 StudyHours(공부 시간) 평균을 구하시오. 
# (단, 결측치는 제거하고, 정답은 소수점 둘째 자리에서 반올림하여 첫째 자리까지 출력)
# print(train.info())

train = train.dropna(subset=['StudyHours'])

train['rank'] = train.groupby('Course')['Score'].rank(method='first', ascending=False)

Ans1 = train[train['rank'] <= 5]['StudyHours'].mean()

# print(round(Ans1, 1)) # 답: 52.2


# Q2. 문자열 분해 & 빈도 (explode 활용)
# Tags 컬럼은 쉼표(,)로 구분된 리뷰 키워드들이다. 
# 태그를 단어 단위로 분리(Split)했을 때, **가장 많이 등장한 단어(Tag)**는 무엇인가? (정답은 문자열로 출력)

tag_list = train['Tags'].str.split(",")

# print(tag_list)

exploded_tags = tag_list.explode()

Ans2 = exploded_tags.str.strip().value_counts().index[0]

# print(Ans2) # 답: Hard

# Q3. 시계열 & 조건 필터링
# JoinDate 컬럼을 활용하여 **'수요일(Wednesday)'**에 가입한 학생들의 수를 구하시오. 
# 그중 Score가 80점 이상인 학생의 비율(%)을 구하시오. 
# (비율 = 조건 만족 학생 수 / 수요일 가입 전체 학생 수) 
# (정답은 소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력. 예: 0.123 -> 0.12)

# print(train.info())

train['JoinDate'] = pd.to_datetime(train['JoinDate'])

train['day'] = train['JoinDate'].dt.dayofweek

Ans3 = len(train[(train['day'] == 2) & (train['Score'] >= 80)]) / len(train[train['day'] == 2])

# print(round(Ans3, 2)) # 답: 0.33