import pandas as pd

# 데이터프레임을 데이터베이스의 "테이블"이라고 생각

# 첫 번째 데이터프레임 (사용자 정보)
df1 = pd.DataFrame({
    'ID': [1, 2, 3],                      # 사용자 ID
    '이름': ['홍길동', '김철수', '이영희']   # 사용자 이름
})

# 두 번째 데이터프레임 (점수 정보)
df2 = pd.DataFrame({
    'ID': [2, 3, 4],        # 사용자 ID (df1과 일부만 겹침)
    '점수': [85, 90, 95]    # 각 ID에 대한 점수
})

# ✅ 조인 (merge): 공통된 ID 기준으로 데이터 합치기
# 'how' 옵션에 따라 조인 방식 달라짐 (inner, outer, left, right)
merged = pd.merge(df1, df2, on='ID', how='inner')  # 공통된 ID만 남김
print("조인\n", merged)

# ✅ 위로 붙이기 (행 방향 concat): 두 데이터프레임을 아래로 이어 붙이기
# 구조가 다르기 때문에 빈 칸(NaN) 발생할 수 있음
df_row_concat = pd.concat([df1, df2], axis=0)  # axis=0: 행 기준
print("위로붙이기\n", df_row_concat)

# ✅ 옆으로 붙이기 (열 방향 concat): 인덱스를 기준으로 옆으로 이어 붙이기
# 인덱스 개수가 다르면 NaN이 생길 수 있음
df_col_concat = pd.concat([df1, df2], axis=1)  # axis=1: 열 기준
print("옆으로붙이기\n", df_col_concat)
