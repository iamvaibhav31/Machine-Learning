import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\3. Regressions\\decision_tree_regression_dataset.csv")

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)



x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
y_predict = regressor.predict(x_grid)
plt.scatter(x, y , label="Actual result" , color="red")
plt.plot(x_grid,y_predict , label="predicted result",color="blue")
plt.xlabel("Level's")
plt.ylabel("Salary")
plt.title("Decision tree Regression Result" , fontsize=20)
plt.legend()
plt.show()

print(y_predict)

y_predict_1 = regressor.predict([[6.5]])
print(y_predict_1)

