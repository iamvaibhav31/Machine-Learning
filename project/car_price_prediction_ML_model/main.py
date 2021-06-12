import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
"""
_-_-_-_-_- DATA PREPROCESSING -_-_-_-_-_
"""
data = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\project\\car_price_prediction_ML_model\\car_info.csv")

# print(data.info()) print the info of the dataset like coloumn name , datatype or Dtype of column , total value in each coloumn etc

# column name year having non year value so lets clean it up

data =data[data["year"].str.isnumeric()] 

# typecast all the data into a int in year colomn

data["year"] = data["year"].astype(int)

# price colomn have two type of value "Ask For Price" and numaric so we have to remove the "Ask For Price" value from colomn price 

data = data[data["Price"]!="Ask For Price"]

# price have a vaule in such formate 80,000 so we convert into 80000 and at last we convert the datatype of a column into int  

data["Price"]=data["Price"].str.replace(",","").astype(int)

data["kms_driven"] =  data["kms_driven"].str.split(" ").str.get(0).str.replace(",","")
data = data[data["kms_driven"].str.isnumeric()]
data["kms_driven"] = data["kms_driven"].astype(int)

data = data[~data["fuel_type"].isna()] # ~ is use for excude the row in which fuel type is null

data["name"] = data["name"].str.split(" ").str.slice(0,3).str.join(" ")

data = data.reset_index(drop = True)

# print(data.describe()) to see std , max , min etc in int colomn

# ther r one outlinner that one car price in 85 lakh so we remove that row

data = data[data["Price"]<6e6].reset_index(drop = True)



# ___END___

# model

x = data.drop(columns = "Price")
y = data["Price"]

x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.3 , random_state=1)

# encode the data 
object_of_OHE = OneHotEncoder()
object_of_OHE.fit(x[["name","company","fuel_type"]])
column_trans = make_column_transformer((OneHotEncoder(categories=object_of_OHE.categories_),["name","company","fuel_type"]),remainder="passthrough")


object_of_linearregression= LinearRegression()
pipe = make_pipeline(column_trans,object_of_linearregression)
pipe.fit(x_train,y_train)

print(pipe.predict(x_test))

