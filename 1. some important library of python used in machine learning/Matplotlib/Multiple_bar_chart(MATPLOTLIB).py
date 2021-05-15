import matplotlib.pyplot as plt
import numpy as np
width = 0.1
country = ["USA" , "India" , "Amarica"]
gold=[20,10,6]
silver=[100,50,30]
Bronze=[200,100,60]

bar1 = np.arange(len(country))

bar2 = [i+width for i in bar1]

bar3= [i+width for i in bar2]

plt.bar(bar1,gold,width,color="r")
plt.bar(bar2,silver,width,color="y")
plt.bar(bar3,Bronze,width)
plt.xlabel("Country")
plt.title("No of medals earn by a country")
plt.ylabel("Medals")
plt.xticks(bar1+width,country)
plt.show()
