import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\4. Classification\\1.1 logstic_regression_dataset.csv")
x =  data.iloc[:,:-1].values
y = data.iloc[:,-1].values

x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 0 , test_size = 0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

classifier = LogisticRegression(random_state=0) 
classifier.fit(x_train,y_train)

y_pridict = classifier.predict(scaler.transform([[30,70000]]))
