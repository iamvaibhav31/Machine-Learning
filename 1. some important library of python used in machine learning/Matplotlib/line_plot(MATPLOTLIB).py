import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,10))
x  = np.linspace(-10 , 10 , 50)
y = 2*x+1
y1=2**x+4*x+5
plt.plot(x,y,color="r",label="X and Y" ,markersize=10,linewidth=1)
plt.plot(x,y1,color="y" , label="X and Y1" ,markersize=10,linewidth=1)
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("the graph between x and y axis")
plt.legend(loc=1)
plt.grid(linestyle="-",linewidth=0.99)
plt.show()

