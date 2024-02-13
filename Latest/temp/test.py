import pandas as pd
file=pd.read_csv('Latest/temp/file.csv')
print(file.head())
X=file.drop(columns=['Height'])
print(X.head())