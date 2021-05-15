import matplotlib.pyplot as plt

country = ["USA" , "India" , "Amarica"]
gold=[20,10,6]
silver=[100,50,30]
Bronze=[200,100,60]
plt.bar(country,gold,width=0.5,color="r",label="Gold")
plt.bar(country,silver,width=0.5,bottom=gold,color="y",label="Silver")
plt.bar(country,Bronze,width=0.5,bottom=silver,label="Bronze")
plt.xlabel("Country")
plt.title("No of medals earn by a country")
plt.ylabel("Medals")
plt.legend()
plt.show()
