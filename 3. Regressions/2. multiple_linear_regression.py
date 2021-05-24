import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression



dataset = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\3. Regressions\\multiple_linear_regression_dataset.csv")

x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,-1].values

ooct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[-1])],remainder="passthrough")
x = np.array(ooct.fit_transform(x))

x_train , x_test , y_train , y_test = train_test_split(x, y, random_state=1, test_size=0.2)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_prediction = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_prediction.reshape(len(y_prediction),1),y_test.reshape(len(y_test),1)),1))