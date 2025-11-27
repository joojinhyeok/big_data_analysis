# [3유형] 통계적 가설 검정
import pandas as pd
train = pd.read_csv('csv/train.csv')

# Q4. 상관분석 유의성 검정 (pearsonr)
# Age와 Fare 간의 피어슨 상관계수와 그에 대한 p-value를 구하시오. (단, Age 결측치는 평균으로 대치 후 수행)
# 상관계수를 소수점 셋째 자리까지 출력하시오. (반올림)
# p-value가 0.05보다 작으면 'Yes', 크면 'No'를 출력하시오. (유의한 관계인가?)

# train['Age'] = train['Age'].fillna(train['Age'].mean())
Ans = train[['Age', 'Fare']].corr().iloc[0, 1]

# print(round(Ans, 3)) # 답: 0.092 이므로 'No'

# ========================================================================================================================================

# Q5. 단일표본 T-검정 (기준값 검정)
# 타이타닉 전체 승객의 **평균 나이(Age)**가 30세와 통계적으로 차이가 있는지 검정하시오. (단, Age 결측치는 제거하고 수행)
# **ttest_1samp**를 수행하여 **검정통계량(statistic)**을 구하시오.
# 정답은 소수점 셋째 자리에서 반올림하여 둘째 자리까지 출력하시오.

d_age = train.dropna(subset=['Age'])['Age']

from scipy.stats import ttest_1samp
Ans = ttest_1samp(d_age, 30)

# print(Ans) # TtestResult(statistic=np.float64(-0.5534583115970276), pvalue=np.float64(0.5801231230388639), df=np.int64(713))

# print(round(Ans[0], 2)) # 답: -0.55

# ========================================================================================================================================

# Q6. 로지스틱 회귀분석 (Odds Ratio)
# Survived를 종속변수(Y), Age와 Fare를 독립변수(X)로 하는 **로지스틱 회귀분석(logit)**을 수행하시오.
# Age 변수의 **오즈비(Odds Ratio)**를 구하시오. (힌트: 오즈비 = exp(회귀계수))
# 정답은 소수점 넷째 자리에서 반올림하여 셋째 자리까지 출력하시오. (참고: from statsmodels.formula.api import logit)

from statsmodels.formula.api import logit
import numpy as np

# 1. 로지스틱 회귀 모델 학습
# 문법: logit('종속변수 ~ 독립변수1 + 독립변수2', data=데이터).fit()
model = logit('Survived ~ Age + Fare', data=train).fit()

# 2. Age의 회귀계수(Coefficient) 뽑기
coef_age = model.params['Age']
# print(coef_age) # 이건 그냥 회귀계수임 (-0.0xxx)

# 3. [핵심] 오즈비(Odds Ratio) 계산
# 힌트대로 exp를 씌워준다!
odds_ratio = np.exp(coef_age)

# 4. 출력 (소수점 셋째 자리까지)
Ans = round(odds_ratio, 3)
print(f"Age의 오즈비: {Ans}")