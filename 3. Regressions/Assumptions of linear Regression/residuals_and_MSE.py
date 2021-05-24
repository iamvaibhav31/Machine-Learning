# The second and immportant assumption in linear regression is That the residual and Mean squared error should be less 
from sklearn.metrics import mean_squared_error
y_actual = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
y_predicted = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9]

print("\n")

# calculate residual
print("RESIDUAL (y_predicted - y_predicted)")
for i in range(len(y_actual)):
    print(y_predicted[i]-y_actual[i])

print("\n")

# calculate mean squared errors
# mean squared errors += pow((y_predicted[i]-y_actual[i]),2)  here "i" is 0 to len(y_actual)-1 
print("MEARN SQARED ERROR OF A DATA")
errors = mean_squared_error(y_actual, y_predicted)
print(errors)

