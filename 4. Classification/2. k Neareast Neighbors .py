import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\4. Classification\\2.1 K-nearest neighbors _dataset.csv")

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)

scalar = StandardScaler()
scalar.fit(x_train,x_test)

classifier = LogisticRegression()

pipe = make_pipeline(scalar , classifier)
pipe.fit(x_train,y_train)

print(pipe.predict([[30,70000]]))