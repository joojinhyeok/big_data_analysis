import pandas as pd

df = pd.read_csv('C:/csv/tested.csv')
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())