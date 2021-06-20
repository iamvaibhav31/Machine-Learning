import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\4. Classification\\6.1 decision tree classifier dataset.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


x_train , x_test , y_train , y_test = train_test_split(x,y,train_size=0.2,random_state=0)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(x_train,y_train)


y_predict = classifier.predict(x_test)
print(y_predict)