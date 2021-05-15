import numpy as np
import pandas as pd  # use for data analysis

# series :- Series is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.). The axis labels are collectively called index.

series = pd.Series([1,2,3,4,54,67,8,9,9])
series1 = pd.Series([1,2,3,4,54,67,8,9,9] , index=[100,101,102,103,104,105,106,107,108] )

print(series)
print("\n")
print(series1)
print("\n")
print(series[0:2])
print("\n")
print(series1[100:105])
print("\n")
print(series1[1:5])
print("\n")
s7 = pd.Series(10 , index=[1,2,3,4,5,6,7,8,9])
print(s7)

# math operation

s3 = pd.Series([1,2,3,4,54,67,8,9,9]) 
s4 = pd.Series([1,2,3,4,74,6,8,9,10])

print(s3+s4)
print("\n")
print(s3-s4)

print("\n")
print(s3*s4)
print("\n")
print(s3/s4)
print("\n")

s5 = pd.Series([1,2,3,4,54,67,8,9,9]) 
s6 = pd.Series([1,2,3,4,74,6])

print(s5+s6)
print("\n")
print(s5-s6)
print("\n")
print(s5*s6)
print("\n")
print(s5/s6)


# DATAFRAME
# A Data frame is a two-dimensional data structure, i.e., data is aligned in a tabular fashion in rows and columns.

print(pd.DataFrame([1,2,3,4]))
print("\n")
print(pd.DataFrame([[1,2,3,4],[1,2,3,4]]))
print("\n")
print(pd.DataFrame([[1,2,3,4],[1,2,3,4],[1,2,3,4]]))
print("\n")
print(pd.DataFrame([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]))
print("\n")
print(pd.DataFrame({"apple":[1,3,5,6,7,8]}))
print("\n")
print(pd.DataFrame({"apple":np.arange(1,10)}))
print("\n")
print(pd.DataFrame({"apple":[[np.arange(1,10)],[1,2,3,4],[1,2,3,4],[1,2,3,4]]}))
print("\n")

df3 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"name":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
df4 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"rank":["werwer","eeeedda","faef","ewwer",'ytrty',"ssgsgr","ggsgsrrr"]})
print(pd.merge(df3,df4))
print("\n")
  
df3 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"name":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
df4 = pd.DataFrame({"rollno":[1,2,3,4,5,6],"rank":["werwer","eeeedda","faef","ewwer",'ytrty',"ssgsgr"]})
print(pd.merge(df3,df4))
print("\n")

df3 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"name":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
df4 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"name":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
print(pd.merge(df3,df4 , on="rollno"))
print("\n")

df3 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"name":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
df4 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"name":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
print(pd.merge(df3,df4 , on="rollno" , suffixes=("person A","person B")))
print("\n")

df3 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"name":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
df4 = pd.DataFrame({"class":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"],"rank":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
print(pd.merge(df3,df4 , left_index=True , right_index=True))
print("\n")

df3 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"rank":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
df4 = pd.DataFrame({"rollno":[1,2,3,4,5,8,9],"rank":["werwer","eeeedda","faef","ewwer",'ytrty',"ssgsgr","ggsgsrrr"]})
print(pd.merge(df3,df4,on="rollno",how="outer",indicator=True))
print("\n")
print(pd.merge(df3,df4,on="rollno",how="inner",indicator=True))
print("\n")
print(pd.merge(df3,df4,on="rollno",how="right",indicator=True))
print("\n")
print(pd.merge(df3,df4,on="rollno",how="left",indicator=True))
print("\n")
df3 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"name":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
df4 = pd.DataFrame({"rollno":[1,2,3,4,5,8,9],"rank":["werwer","eeeedda","faef","ewwer",'ytrty',"ssgsgr","ggsgsrrr"]})
print(pd.merge(df3,df4,on="rollno",how="outer",indicator=True))
print("\n")

df3 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"name":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
df4 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"rank":["werwer","eeeedda","faef","ewwer",'ytrty',"ssgsgr","ggsgsrrr"]})
print(pd.concat([df3,df4]))
print("\n")

df3 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"name":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
df4 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"rank":["werwer","eeeedda","faef","ewwer",'ytrty',"ssgsgr","ggsgsrrr"]})
print(pd.concat([df3,df4],ignore_index=True))
print("\n")
print(pd.concat([df3,df4],axis=0))
print("\n")
print(pd.concat([df3,df4],axis=1))
print("\n")

df3 = pd.DataFrame({"rollno":[1,2,3,4,5,6,7],"name":["vai","aman","awawaw","dadaa","fefeaf",'eaefef',"tery"]})
df4 = pd.DataFrame({"rollno":[1,2,3,4,5],"rank":["werwer","eeeedda","faef","ewwer",'ytrty']})
print(pd.concat([df3,df4]))
print("\n")
print(pd.concat([df3,df4],ignore_index=True))
print("\n")
print(pd.concat([df3,df4],keys=["data frame 2 " , "data frame 4 "]))
print("\n")

df3 = pd.DataFrame({"one":[1,2,3],"two":[4,5,6]})
df4 = pd.DataFrame({"three":[7,8,9],"four":[10,11,12]})
print(df3.append(df4))
print("\n")
print(df3.append(df4,ignore_index=True))
print("\n")
print(df3.join(df4))
print("\n")

df= pd.read_csv("C:/Users/evil1/Desktop/some important library of python used in machine learning/panda/Data.csv")
print(df)
print("\n")
print(df.head()) # print first five value from data
print("\n")
print(df.tail()) # print last five value from data
print("\n")
print(df.columns) # print name of coloumn
print("\n")
print(df.isnull()) # print the true if there is null value or print false when the value is no null in data but no in original data set 
print("\n")
print(df.notnull()) # oposite to isnull
print("\n")
print(df.isnull().sum()) # print the total value in column
print("\n")
print(df.notnull().sum()) # oposite to isnull.sum
print("\n")
print(df.isnull().sum().sum()) # print the total null value present in data
print("\n")
print(df.notnull().sum().sum()) # oposite to isnull.sum.sum
print("\n")
print(df.dropna()) # print that row which did not have a null value
print("\n")
print(df.dropna(how="any")) # same to df.dropna
print("\n")
print(df.dropna(how="all")) # all is use to remove that row comtain only null value
print("\n")
#print(df.dropna(subset=['salary'])) # subset is used when u want to remove row of null value from specific column
print("\n")
print(df.dropna(inplace=False)) 
print("\n")
print(df.dropna(inplace=True))# similar to  df.dropna
print("\n")
print(df.fillna(200))
print("\n")
df1= pd.read_csv("C:/Users/evil1/Desktop/some important library of python used in machine learning/panda/Data.csv")
print(df1)
print("\n")
print(df1.fillna(method="bfill",limit=1))
print("\n")
print(df1.fillna(method="bfill"))
print("\n")
df2= pd.read_csv("C:/Users/evil1/Desktop/some important library of python used in machine learning/panda/Data.csv")
print(df2.fillna(method="ffill",limit=1))
print("\n")
print(df2.fillna(method="ffill"))
print("\n")
df2= pd.read_csv("C:/Users/evil1/Desktop/some important library of python used in machine learning/panda/Data.csv")
print(pd.melt(df2))
print("\n")
print(pd.melt(df2,id_vars=["Age"]))
