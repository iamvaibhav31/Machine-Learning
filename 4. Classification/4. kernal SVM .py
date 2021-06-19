from operator import imod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



data = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\4. Classification\\4.1 kernql svm_dataset.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values



x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)



scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(x_train,y_train)



y_predict = classifier.predict(x_test)



print(y_predict)