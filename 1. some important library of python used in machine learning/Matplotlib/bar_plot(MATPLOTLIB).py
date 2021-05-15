import matplotlib.pyplot as plt

country = ["USA" , "India" , "Amarica"]
gold=[20,10,6]

plt.bar(country,gold,width=0.5,color="r",edgecolor="y",linewidth=4,linestyle="--",alpha=0.96)
plt.xlabel("Country")
plt.title("No of medals earn by a country")
plt.ylabel("Medals")
plt.show()
