import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\3. Regressions\\polynomial_linear_regression_dataset.csv")

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# Train the simple linear regression modal

lin_regressor = LinearRegression()
lin_regressor.fit(x, y)

# Train the polynomial linear regression modal

poly = PolynomialFeatures(degree=9)
x_poly = poly.fit_transform(x)
poly_regressor = LinearRegression()
poly_regressor.fit(x_poly,y)

# visuallising the result of simple linear regression modal
y_lin_predist = lin_regressor.predict(x)
plt.scatter(x, y , label="Actual result" , color="red")
plt.plot(x,y_lin_predist , label="predicted result",color="blue" ,marker="o" , markersize=4)
plt.xlabel("Level's")
plt.ylabel("Salary")
plt.title("Linear Regression Result" , fontsize=20)
plt.legend()
plt.show()

# visuallising the result of polynomial linear regression modal
y_poly_predist = poly_regressor.predict(x_poly)
plt.scatter(x, y , label="Actual result" , color="red")
plt.plot(x,y_poly_predist , label="predicted result",color="blue" ,marker="o" , markersize=4)
plt.xlabel("Level's")
plt.ylabel("Salary")
plt.title("Polynomial Regression Result" , fontsize=20)
plt.legend()
plt.show()

np.set_printoptions(precision=2)
# predicting the result of Level 6.5 from Simple Linear regression modal
y_lin_predist_1 = lin_regressor.predict([[6.5]])
print(y_lin_predist_1)
print("\n")
# predicting the result of Level 6.5 from Polynomial Linear regression modal
y_poly_predist_1 = poly_regressor.predict(poly.fit_transform([[6.5]]))
print(y_poly_predist_1)
