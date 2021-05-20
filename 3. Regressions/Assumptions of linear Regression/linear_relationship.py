# The first assumption in linear regression is A Dataset Should In Linear Relationship Form
from matplotlib import markers
from matplotlib.markers import MarkerStyle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\3. Regressions\\Assumptions of linear Regression\\dataset.csv")

x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# the linear relationship mean that the scatter plot between X and Y in a strate line having a slope M

plt.scatter(x, y)
plt.xlabel(" Year Of Experience " )
plt.ylabel(" Salary ")
plt.grid()
plt.show()