#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT 1

# Pima Indians Diabetes Database.
# It consists 768 tuples each having 9 attributes.

# In[1]:


import pandas as pd
#imported the dataset given to us
pid=pd.read_csv("C:\\Users\\Micontroller Lab N16\\IIT MANDI\\3rd Week\\pima-indians-diabetes.csv",sep=',')
#made a copy of the original dataset
pid1=pid.copy()
print(pid1)
#looked for any null value present in the dataset
print(pid1.isnull().sum())


# In[2]:


pid.columns
#created a list of all the columns present in the dataset
pid_col=list(pid.columns)
pid_col
pid_col1=pid_col.copy()
#we do not want to bring changes in the class column so we removed it from the copied dataset
pid_col1.remove('class')
pid_col1


# Question 1. Write a python program to
# 
# a. Normalize all the attributes, except class attribute, of pima-indians-diabetes.csv
# using min-max normalization to transform the data in the range [0-1]. Save the file as
# pima-indians-diabetes-Normalised.csv
# 
# b. Standardize, all the attributes, except class attribute, of pima-indians-
# diabetes.csv using z-normalization. Save the file as pima-indians-
# diabetes-Standardised.csv

# In[3]:


#Using minmaxscaler function from scikit-learn library for min max normalization
from sklearn.preprocessing import MinMaxScaler
pid1_mms=pid.copy() 
scaler = MinMaxScaler(copy=True, feature_range=(0,1))
print(scaler.fit(pid1_mms))                                     #fitting the model
pid1_mms=scaler.fit_transform(pid1_mms)
print(pid1_mms)                                                  #normalized dataset
print('\n\n')


# In[4]:


#Using StandardScaler function from scikit-learn library for standardization.
from sklearn.preprocessing import StandardScaler
pid1_ss=pid1.copy() 
scaler = StandardScaler()
standard_df = scaler.fit_transform(pid1_ss)                      #fitting the model
print(pid1_ss)                                                   #standardized dataset


# Question 2. Split the data of each class from pima-indians-diabetes.csv into train data and test
# data. Train data contain 70% of tuples from each of the class and test data contain remaining
# 30% of tuples from each class. Save the train data as diabetes-train.csv and save the
# test data as diabetes-test.csv
# a. Classify every test tuple using K-nearest neighbor (KNN) method for the different values
# of K (1, 3, 5, 7, 9, 11, 13, 15, 17, 21). Perform the following analysis :
# i. Find confusion matrix (use ‘confusion_matrix’) for each K.
# ii. Find the classification accuracy (You can use ‘accuracy_score’) for each K. Note the
# value of K for which the accuracy is high.

# In[5]:


X1 = pid1.drop(['class'], axis = 1)# X denotes the input functions and here class defines whether the person is ill or not
print(X1)
y1 = pid['class']                  #y denotes the output functions
print(y1)


# In[6]:


from sklearn.model_selection import train_test_split       #As given we are assigning 70% of data for training and 30% for testing
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size = 0.7, random_state = 42)

print(X1_train)
print(y1_train)
print(X1_test)
print(y1_test)


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
neighbors=[1,3,5,7,9,11,13,15,17,19,21]
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
  
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X1_train, y1_train)
    print('Predicted Outcomes for neighbours =',k,'are', knn.predict(X1_test))
    print('\n')
    print('Accuracy = ',knn.score(X1_test, y1_test))
    print('\n')
    matrix = confusion_matrix(y1_test,knn.predict(X1_test))
    print('Confusion Matrix = ',matrix)
    print('\n\n')
    
      
    # Compute traning and test data accuracy
    train_accuracy[i] = knn.score(X1_train, y1_train)
    test_accuracy[i] = knn.score(X1_test, y1_test)
    
  
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')


  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# Question 3. Split the data of each class from pima-indians-diabetes-Normalised.csv into
# train data and test data. Train data should contain same 70% of tuples in Question 2 from
# each of the class and test data contain remaining same 30% of tuples from each class. Save the
# 
# train data as diabetes-train-normalise.csv and save the test data as diabetes-
# test-normalise.csv
# 
# a. Classify every test tuple using K-nearest neighbor (KNN) method for the different values
# of K (1, 3, 5, 7, 9, 11, 13, 15, 17, 21). Perform the following analysis :
# i. Find confusion matrix (use ‘confusion_matrix’) for each K.
# ii. Find the classification accuracy (You can use ‘accuracy_score’) for each K. Note the
# value of K for which the accuracy is high.

# In[8]:


X2 = pid1_mms
X2
y2 =  pid['class']
y2


# In[9]:


from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, train_size = 0.7, random_state = 42)

print(X2_train)
print(y2_train)
print(X2_test)
print(y2_test)


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
neighbors=[1,3,5,7,9,11,13,15,17,19,21]
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
acc=[]
  
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X2_train, y2_train)
    print('Predicted Outcomes for neighbours =',k,'are', knn.predict(X2_test))
    print('\n')
    print('Accuracy = ',knn.score(X2_test, y2_test))
    print('\n')
    matrix = confusion_matrix(y2_test,knn.predict(X2_test))
    print('Confusion Matrix = ',matrix)
    print('\n\n')
    
      
    # Compute traning and test data accuracy
    train_accuracy[i] = knn.score(X2_train, y2_train)
    test_accuracy[i] = knn.score(X2_test, y2_test)
    
  
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')


  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# Question 4. Split the data of each class from pima-indians-diabetes-Standardised.csv into
# train data and test data. Train data should contain same 70% of tuples in Question 2 from
# each of the class and test data contain remaining same 30% of tuples from each class. Save the
# 
# train data as diabetes-train-standardise.csv and save the test data as diabetes-
# test-standardise.csv
# 
# a. Classify every test tuple using K-nearest neighbor (KNN) method for the different values
# of K (1, 3, 5, 7, 9, 11, 13, 15, 17, 21). Perform the following analysis :
# i. Find confusion matrix (use ‘confusion_matrix’) for each K.
# 
# ii. Find the classification accuracy (You can use ‘accuracy_score’) for each K. Note the
# value of K for which the accuracy is high.

# In[11]:


X3 = pid1_ss
X3
y3 = pid1_ss['class']
y3


# In[12]:


from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, train_size = 0.7, random_state = 42)

print(X3_train)
print(y3_train)
print(X3_test)
print(y3_test)


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
neighbors=[1,3,5,7,9,11,13,15,17,19,21]
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
acc=[]
  
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X3_train, y3_train)
    print('Predicted Outcomes for neighbours =',k,'are', knn.predict(X3_test))
    print('\n')
    print('Accuracy = ',knn.score(X3_test, y3_test))
    print('\n')
    matrix = confusion_matrix(y3_test,knn.predict(X3_test))
    print('Confusion Matrix = ',matrix)
    print('\n\n')
    
      
    # Compute traning and test data accuracy
    train_accuracy[i] = knn.score(X3_train, y3_train)
    test_accuracy[i] = knn.score(X3_test, y3_test)
    
  
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')


  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# 6. Why the value of K is considered as odd integer?

# Suppose P1 is the point, for which label needs to predict. First, you find the k closest point to P1 and then classify points by majority vote of its k neighbors. Each object votes for their class and the class with the most votes is taken as the prediction.
# if k=even it will be difficult to choose class as the voting may be tied.
