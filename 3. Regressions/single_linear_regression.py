import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\3. Regressions\\Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.2 ,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predictting the result of training ond test set 

y_train_predict= regressor.predict(x_train)
y_test_predict = regressor.predict(x_test)

# Visualisation the Test set result

plt.scatter(x_train,y_train,color="red",label="Actual Data Of Training Set")
plt.plot(x_train,y_train_predict,color="blue",label="Predicted Data Of Training Set")
plt.title(" Salary Vs Experience (Test set) ")
plt.xlabel("Year Of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Visualisation the training set result

plt.scatter(x_test,y_test,color="red",label="Actual Data Of Test Set")
plt.plot(x_test,y_test_predict,color="blue",label="Predicted Data Of Test Set")
plt.title(" Salary Vs Experience (Training set) ")
plt.xlabel("Year Of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()