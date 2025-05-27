import pandas as pd

df = pd.read_csv('C:/csv/test.csv')
print(df.shape)
print(df.info())
print(df.describe())
print("isnull().sum()", df.isnull().sum())