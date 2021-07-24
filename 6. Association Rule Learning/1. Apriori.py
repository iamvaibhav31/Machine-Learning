import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori


def inspect(result):
    lhs = [tuple(result[2][0][0])[0] for result in result]
    rhs = [tuple(result[2][0][1])[0] for result in result]
    support = [result[1] for result in result]
    confidence = [result[2][0][2] for result in result]
    lifts = [result[2][0][3] for result in result]
    return list(zip(lhs,rhs,support,confidence,lifts))


data = pd.read_csv("C://Users//evil1//Desktop//Machine Learning//6. Association Rule Learning//Market_Basket_Optimisation.csv",header=None)


transaction_lst = []
for i in range(0,7501):
    transaction_lst.append([str(data[i,j]) for j in range(0,20)])


rules = apriori(transactions=transaction_lst,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2 ,max_length=2)


result = list(rules)
result_data = pd.DataFrame(inspect(result),columns=["Left Hand Side","Right Hand Sid","Support","Confidence","Lift"]) 

print(result_data)
 