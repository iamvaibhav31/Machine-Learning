import numpy as np
import random


arr = np.array([1,2,3]) # to create one dimention array

print(arr)

print(arr.ndim) # to find out the dimenttion of an array

print(arr.shape)# find out the shape of the arrary means row and collumn

print(arr.size) # print the size of an arrary

arr2 = np.array([[1,53,3,2,4],[1,3,6,7,8]])

print(arr2)

print(arr2.ndim) 

print(arr2.shape)

print(arr2.size)

arr3 = np.arange(1,30)# create the array in between the range 

print(arr3)

arr4 = np.arange(1,40,2) # create the array in between the range but step the array by two

print(arr4)

arr5 = np.linspace(1,10,5) # The numpy.linspace() function returns number spaces evenly w.r.t interval. Similar to numpy.arange() function but instead of step it uses sample number.

print(arr5)

arr6 = np.linspace(1,100,5)

print(arr6)

arr7 = np.logspace(1,2,5) #  in logspace starting two parameter 1 and 2 are use in the form of 10 ki power and the out is similar to linespace

print(arr7)

arr8 = np.arange(1,17).reshape(4,4) # create multi dimension array

print(arr8)

print(arr8.ravel()) # convert multi dimension array into single dimension

print(arr2.flatten())

print(arr2.transpose()) 

arr9 = np.arange(17,33).reshape(4,4)

# MATH

print(arr8+arr9)

print(arr8-arr9)

print(arr8*arr9)

print(arr8/arr9)

print(arr8@arr9) # use for row column multiplication

print(np.sum(arr9))

print(np.mean(arr9))

print(np.sqrt(arr9))

print(np.log(arr9))

print(np.std(arr9))

# SLICING

arr10 = np.arange(1,101).reshape(10,10)

print(arr10)

print(arr10[0][9])

print(arr10[:]) # give the full element

print(arr10[::]) # give the full element

print(arr10[:,1])  # print the first column

print(arr10[:,0:1])

print(np.sin(arr8))

print(np.sin(180))


# concatination

print(np.concatenate((arr8,arr9))) # add in a row

#print(np.vstack(arr8,arr9))

print(np.concatenate((arr8,arr9),axis=1)) # add in a coloumn

#print(np.hstack(arr8,arr9))

print(np.random.random(1)) #giving the random variable in b/w 0 or 1 having a float data 

print(np.random.randint(2,9))

print(np.random.random((3,3)))

print(np.random.randint(1,5,(3,3)))

print(np.random.rand(9))

print(np.ones(9,dtype=int))

print(np.ones((9,9),dtype=int))

print(np.zeros(9,dtype=int))

print(np.zeros((9,9),dtype=int))

# string operation 

fname = "vaibhav"
sname = "sharma"

print(np.char.upper(fname))
 
print(np.char.lower(fname))

print(np.char.center(fname,50))

print(np.char.center(fname,100,fillchar="#"))

