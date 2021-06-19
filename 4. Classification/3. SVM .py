from numpy.ma import arange
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


data = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\4. Classification\\3.1 SVM_dataset.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

x_train ,x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

classifier = SVC(kernel="linear" , random_state=0)
classifier.fit(x_train,y_train)

y_predict = classifier.predict(scaler.transform([[30,90000]]))
y_prediction = classifier.predict(x_test)

x_set , y_set = scaler.transform(x_train) , y_train
x1 , x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-10,stop=x_set[:,1].max()+10,step = 1),
                        np.arange(start=x_set[:,1].min()-1000,stop = x_set[:,1].max()+1000 ,step=1))
plt.contourf(x1,x2,classifier.predict(scaler.transform(np.array([x1.ravel(),x2.ravel()]).T)).reshape(x1.shape),alpha=0.79,cmap=ListedColormap(("red","green")))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i , j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(("red","green"))(i),label=j)
plt.title("SVM (Traing set)")
plt.xlabel("Age")
plt.ylabel("Estimated salary")
plt.legend()
plt.show()