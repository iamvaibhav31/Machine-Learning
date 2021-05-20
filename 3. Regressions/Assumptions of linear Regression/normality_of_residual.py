import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

x_actual = np.array([2,1,4,9,9,7,5,1,9,6,10,6,4])
y_actual = np.array([0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7])

x_actual = x_actual.reshape(-1,1)
y_actual = y_actual.reshape(-1,1)

model = LinearRegression()
model.fit(x_actual,y_actual)
y_predict = model.predict(x_actual)
mse  =  mean_squared_error(y_actual,y_predict)

y_residual=[]
result = 0
for i in range(len(y_actual)):
    result = y_predict[i][0] - y_actual[i][0]
    y_residual.append(result)

y_residual = np.array(y_residual)
y_residual = y_residual.reshape(-1,1)
print(y_residual)

plt.hist(y_residual)
plt.show()