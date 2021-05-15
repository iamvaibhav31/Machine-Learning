import numpy as np
# numpy is use for play with array
import matplotlib.pyplot as plt
# matplotlib is use for ploating graphe for perticulare data
import pandas as pd
#

# IMPORTING THE DATASET 
 
dataset = pd.read_csv("Data.csv")
# dataset a variable that contain all the data of ML Modal
# read_csv thiis function able to read a file having a extantion .csv
matix_of_feature = dataset.iloc[:,:-1].values
# Matric of feature is also called independed variable through which we can predict the depended variable
# iloc is a function of pandas library through which we can select the row (":" means selcect all the row) or column (":-1" means select all the column except last column)
# values is use to assing the  value of selected row and column to matrix of feature a variable
depending_variable_vector = dataset.iloc[:,-1].values
# depending variable vector is depend on matrix of feature all the ML modals are made on the bass of depending variable vector

print(matix_of_feature)
print("\n")
print(depending_variable_vector)
print("\n")
print(dataset) 

# TAKING CARE OF MISSING DATA IN DATASET

from  sklearn.impute import SimpleImputer 
# sklearn is best library for data scientist 

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
# SimpleImputer is a class found in package sklearn. impute. It is used to impute / replace the numerical or categorical missing data related to one or more features with appropriate values such as following: Each of the above type represents strategy when creating an instance of SimpleImputer
# The missing_values placeholder which has to be imputed. By default is NaN
# strategy is a what operation is going to perform will replacing the np.nan or none or float or int from the data set
imputer.fit(matix_of_feature[:,1:3])
matix_of_feature[:,1:3] = imputer.transform(matix_of_feature[:,1:3])
# fit(X, y) :- Learns about the required aspects of the supplied data and returns the new object with the learned parameters. It does not change the supplied data in any way. transform() :- Actually transform the supplied data to the new form.
# fit to connect the imputer to matix_of_feature
# transform is apply funtion to matix_of_feature
print("\n")
print(matix_of_feature)

# ENCODING CATEGORIAL DATA

###_ONE_HOT_ENCODER_###
# convert the data into bainary vector 
#  example in this data set we have three repeted country name in country column france spain and germany  are convertd into [1.0 0.0 0.0] [0.0 0.0 1.0] and [0.0 1.0 0.0] this bainary vector or country column is divided in to three part in order of having 3 repeted name if there are 4 repeled name in that case country coulmn is divided in 4 parts
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , LabelEncoder

object_of_columnTransformer =  ColumnTransformer(transformers = [('encoder' , OneHotEncoder() , [0])] , remainder='passthrough')
matix_of_feature = np.array(object_of_columnTransformer.fit_transform(matix_of_feature))
print("\n")
print(matix_of_feature)

###_LABEL_ENCODER_###
# convert into  0 or 1
# lable encodder is only done when column only have 2 repeted value like yess or no or true or false  then no or false are converted into 0 and yes or true converted in to 1
object_of_LabelEncoder = LabelEncoder()
depending_variable_vector = object_of_LabelEncoder.fit_transform(depending_variable_vector)
print("\n")
print(depending_variable_vector)

# SPLITTING THE DATASET INTO TRAINING SET AND TEST SET
# traning set is to train the ML modal on excicting observation
# test set is to evaluate the proformance of your ML modal to new observation(future or new data)

from sklearn.model_selection import train_test_split

matix_of_feature_training , matix_of_feature_test , depending_variable_vector_traning , depending_variable_vector_test = train_test_split(matix_of_feature,depending_variable_vector,test_size=0.2,random_state=1)
print("\n")
print(matix_of_feature_training)
print("\n")
print(matix_of_feature_test)
print("\n")
print(depending_variable_vector_traning)
print("\n")
print(depending_variable_vector_test)

# FEATURE SCALING

# feature scaling which are allow to put the feature of ML modal in same scale to prevent the 
# feature over loading mean that the feature that are overloaded hat feature is not allow in your Ml model

# to perform the feature scaling on your data there are two methord
# 1 . Standardisation
# 2 . Normalisation

# Standardisation

from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()
matix_of_feature_training[:,3:] = Scaler.fit_transform(matix_of_feature_training[:,3:])
matix_of_feature_test[:,3:] = Scaler.transform(matix_of_feature_test[:,3:])

print("\n")
print(matix_of_feature_training)
print("\n")
print(matix_of_feature_test)
print("\n")

# Normalisation

from sklearn.preprocessing import MinMaxScaler

Scaler1 = MinMaxScaler()
matix_of_feature_training[:,3:] = Scaler1.fit_transform(matix_of_feature_training[:,3:])
matix_of_feature_test[:,3:] = Scaler1.transform(matix_of_feature_test[:,3:])

print("\n")
print(matix_of_feature_training)
print("\n")
print(matix_of_feature_test)
print("\n")
