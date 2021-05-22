import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
dataset = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\3. Regressions\\support_vector_regresson_dataset.csv")

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)
np.set_printoptions(precision=2)
sc_x = StandardScaler()
X = sc_x.fit_transform(x)
sc_y= StandardScaler()
Y = sc_y.fit_transform(y)

regressor = SVR(kernel="rbf") # RBF = 
regressor.fit(X,Y)
y_prediction = sc_y.inverse_transform(regressor.predict(X))

plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(Y), label="Actual result" , color="red")
plt.plot(sc_x.inverse_transform(X),y_prediction , label="predicted result",color="blue" ,marker="o" , markersize=4)
plt.xlabel("Level's")
plt.ylabel("Salary")
plt.title("SVR Regression Result" , fontsize=20)
plt.legend()
plt.show()

y_prediction_1 = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
print(y_prediction_1)
