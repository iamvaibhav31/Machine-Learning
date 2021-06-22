from joblib.logger import PrintTime
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\4. Classification\\7.1 random tree classifier dataset.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)


scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform( x_test )


classifier = RandomForestClassifier(n_estimators=12 , criterion="entropy",random_state=0)
classifier.fit(x_train,y_train)


y_predict = classifier.predict(x_test)
print(y_predict)