import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
dataset = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\3. Regressions\\multiple_linear_regression_dataset.csv")

x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,-1].values

ooct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[-1])],remainder="passthrough")
x = np.array(ooct.fit_transform(x))

x_train , x_test , y_train , y_test = train_test_split(x, y, random_state=1, test_size=0.2)

print(x_train)
print("\n")
print(x_test)
print("\n")
print(y_train)
print("\n")
print(y_test)