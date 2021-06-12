from os import pipe
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder ,StandardScaler
from sklearn.linear_model import LinearRegression , Lasso , Ridge
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


data = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\project\\Bangalore House Price Prediction\\Bengaluru_House_Data.csv")

def convertrange(values):
    try:
        temp = values.split("-")
        if len(temp) == 2:
            return (float(temp[0])+float(temp[1]))/2
        else:
            return float(temp[0])
    except:
        return None

def removepricepersquarefeetoutline(df):
    df_output = pd.DataFrame()
    for key , subdf in df.groupby("location"):
        mean = np.mean(subdf.price_per_square_feet)
        standerdeviation = np.std(subdf.price_per_square_feet)
        gen_df = subdf[(subdf.price_per_square_feet  >= (mean-standerdeviation))&(subdf.price_per_square_feet  <= (mean+standerdeviation))]
        df_output = pd.concat([df_output,gen_df],ignore_index=True)
    return df_output

def removebhkoutliner(df):
    exclude_indices = np.array([])
    for location_key,location_subdf in df.groupby("location"):
        bhk_stats={}
        for bhk_key,bhk_subdf in df.groupby("BHK"):
            bhk_stats[bhk_key]={
                "mean":np.mean(bhk_subdf.price_per_square_feet),
                "standerdeviation":np.std(bhk_subdf.price_per_square_feet),
                "count":bhk_subdf.shape[0]
            }
        for bhk_key_1 , bhk_subdf_1 in df.groupby("BHK"):
            stats = bhk_stats.get(bhk_key_1-1)
            if stats and stats["count"]>=5:
               exclude_indices = np.append(exclude_indices,bhk_subdf_1[bhk_subdf_1.price_per_square_feet<(stats["mean"])].index.values) 
        return df.drop(exclude_indices,axis = "index")


"""for column in data.columns:
    print(data[column].value_counts())
    print("_#_"*20)
    print("#_#"*20)""" # to see how many time value repeted
# print(data.isna().sum()) # to see total null values in each column
data.drop(["area_type","society","balcony"], axis = 1, inplace = True)
"""print(data.describe()) to check the out liner of the data
print(data.info())
"""
data["location"] = data["location"].fillna("Thanisandra")
data["size"] = data["size"].fillna("2 BHK")
data["bath"] = data["bath"].fillna(data["bath"].median())
data["BHK"] = data["size"].str.split(" ").str.get(0).astype(int)
#print(data["total_sqft"].unique()) to check the unique value inn a column
data["total_sqft"] = data["total_sqft"].apply(convertrange) 
data["total_sqft"] = data["total_sqft"].fillna(data["total_sqft"].median())
data["price_per_square_feet"] = data["price"]*100000 / data["total_sqft"]
data["location"] = data["location"].apply(lambda x: x.strip())
locationcount = data["location"].value_counts()
locataionlessthan10 = locationcount[locationcount<10]
data["location"] = data["location"].apply(lambda x : " other" if x in locataionlessthan10 else x)
data = data[(data["total_sqft"]/data["BHK"])>=300]
data = removepricepersquarefeetoutline(data)
data = removebhkoutliner(data)
data.drop(columns=["size","price_per_square_feet"],inplace=True)
#data.to_csv("datapreprocessing_house_data.csv")

x = data.drop(columns=["price"])
y = data["price"]

x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size=0.3 ,random_state=0)

column_trans = make_column_transformer((OneHotEncoder(sparse=False),["location"]),remainder="passthrough")
scaler = StandardScaler()

lrm = LinearRegression(normalize=True)
pipe_lrm = make_pipeline(column_trans,scaler,lrm)
pipe_lrm.fit(x_train,y_train)

lm = Lasso()
pipe_lm = make_pipeline(column_trans,scaler,lm)
pipe_lm.fit(x_train,y_train)

rm = Ridge()
pipe_rm = make_pipeline(column_trans,scaler,rm)
pipe_rm.fit(x_train,y_train)
print("\n")
print(" Linear Regression Model")
print("\n")
y_predict_lrm = pipe_lrm.predict(x_test)
print("\n")
print("Lasso Modal")
print("\n")
y_predict_lm = pipe_lm.predict(x_test)
print("\n")
print("Ridge Model")
print("\n")
y_predict_rm = pipe_rm.predict(x_test)
print("\n")