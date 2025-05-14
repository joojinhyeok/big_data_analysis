import pandas as pd

# ✅ 주문 데이터 (order 정보)
orders = pd.DataFrame({
    'user_id': [1, 2, 1, 3],               # 사용자 ID
    'product': ['A', 'B', 'A', 'C'],       # 구매한 상품
    'amount': [2, 1, 1, 5]                 # 구매 수량
})

# ✅ 사용자 정보 (성별 포함)
users = pd.DataFrame({
    'user_id': [1, 2, 3],                  # 사용자 ID
    'gender': ['F', 'M', 'F']              # 성별
})

# ✅ 두 데이터프레임을 user_id 기준으로 조인 (merge)
# orders에 gender 정보를 붙이는 과정
merged = pd.merge(orders, users, on='user_id')

# ✅ 피벗 테이블 생성
# 성별(gender)을 행으로, 상품(product)을 열로
# 구매 수량(amount)의 총합을 집계
# 구매하지 않은 경우는 0으로 채움 (fill_value=0)
pivot = pd.pivot_table(
    merged,
    values='amount',             # 집계할 값 (구매 수량)
    index='gender',              # 행 인덱스 (성별)
    columns='product',           # 열 인덱스 (상품 종류)
    aggfunc='sum',               # 집계 함수: 합계
    fill_value=0                 # NaN 대신 0으로 채움
)

# ✅ 최종 결과 출력
print(pivot)
