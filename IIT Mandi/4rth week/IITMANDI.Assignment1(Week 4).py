#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt


# In[2]:


import pandas as pd
#imported the dataset given to us
pid=pd.read_csv("C:\\Users\\Micontroller Lab N16\\IIT MANDI\\3rd Week\\pima-indians-diabetes.csv",sep=',')

#made a copy of the original dataset
pid1=pid.copy()
print(pid1)


# In[3]:


#pid.columns
#created a list of all the columns present in the dataset
#pid_col=list(pid.columns)
pid2=pid.copy()
#pid_col
#pid_col1=pid_col.copy()
#we do not want to bring changes in the class column so we removed it from the copied dataset
#pid_col1.remove('class')
pid2.drop(["class"], axis = 1, inplace = True)

print(pid2)


# ## Ques 1 Show the performance of K-nearest neighbor (KNN) classifier for different values of K (1, 3, 5, 7, 9,
# ## 11, 13, 15, 17, 19, 21)
# 
# ## A. Find confusion matrix (use ‘confusion_matrix’) for each K.
# 
# ## B. Find the classification accuracy (You can use ‘accuracy_score’) for each K. Note the value of K for
# ## which the accuracy is high.

# In[4]:


X1 = pid2                  # X denotes the input functions and here class defines whether the person is ill or not
print(X1)
y1 = pid['class']                  #y denotes the output functions
print(y1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, 
                                                    train_size=0.7, 
                                                    random_state=42,
                                                    stratify=y1)

print(X1_train)
print(X1_test)
print(y1_train)
print(y1_test)

print(f"Numbers of train instances by class: {np.bincount(y1_train)}")
print(f"Numbers of test instances by class: {np.bincount(y1_test)}")


# ![train_test_split.jpg](attachment:train_test_split.jpg)

# In[5]:


X1_test, X1_val, y1_test, y1_val = train_test_split(X1_test, y1_test, 
                                                    train_size=0.5, 
                                                    random_state=42,
                                                    stratify=y1_test)
print(X1_train)
print(X1_val)
print(y1_test)
print(y1_val)

print(f"Numbers of test instances by class: {np.bincount(y1_test)}")
print(f"Numbers of validation instances by class: {np.bincount(y1_val)}")


# In[6]:


from sklearn.neighbors import KNeighborsClassifier

neighbors=[1,3,5,7,9,11,13,15,17,19,21]
train_accuracy = np.empty(len(neighbors))
val_accuracy = np.empty(len(neighbors))
acc=[]
  
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X1_train, y1_train)
    print('Predicted Outcomes for neighbours =',k,'are', knn.predict(X1_val))
    print('\n')
    
    print('Accuracy = ',knn.score(X1_val, y1_val))
    if ((knn.score(X1_val, y1_val))>=0):
        acc.append(knn.score(X1_val, y1_val))
    print('\n')
   
    matrix = confusion_matrix(y1_val,knn.predict(X1_val))
    print('Confusion Matrix = ',matrix)
    print('\n\n')
    
      
    # Compute traning and test data accuracy
    train_accuracy[i] = knn.score(X1_train, y1_train)
    val_accuracy[i] = knn.score(X1_val, y1_val)

print(acc)
print('\n')
print('maximum accuracy is =', max(acc)*100)

  
# Generate plot
plt.plot(neighbors, val_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')


plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[7]:


from sklearn.neighbors import KNeighborsClassifier

neighbors=[1,3,5,7,9,11,13,15,17,19,21]
#train_accuracy = np.empty(len(neighbors))
#val_accuracy = np.empty(len(neighbors))
acc=[]
  
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X1_train, y1_train)
    print('Predicted Outcomes for neighbours =',k,'are', knn.predict(X1_test))
    y_pred=knn.predict(X1_test)
    y_pred
    
    print('Accuracy = ',knn.score(X1_test, y1_test))
    if ((knn.score(X1_test, y1_test))>=0):
        acc.append(knn.score(X1_test, y1_test))
    print('\n')
    
    matrix = confusion_matrix(y1_test,knn.predict(X1_test))
    print('Confusion Matrix = ',matrix)
    
    print('\n')
    
print(acc)
print('\n')
print('maximum accuracy is =', max(acc)*100)


# ## Ques.2 Build a Bayes classifier with Multi-modal Gaussian distribution (GMM) with Q components (modes)as class conditional density for each class. Show the performance for different values of Q (2, 4, 8, 16).
# ## Estimate the parameters of the Gaussian Mixture Model (mixture coefficients, mean vectors and covariance matrices) using maximum likelihood method.
# ## A. Find confusion matrix (use ‘confusion_matrix’) for each Q.
# ## B. Find the classification accuracy (You can use ‘accuracy_score’) for each Q.
# ## C. Observe the values in the covariance matrix in each case and comment.
# ## D. Compare the results with that obtained using Bayes classifier with unimodal Gaussian distribution
# ## (Q = 1).

# Clustering methods such as K-means have hard boundaries, meaning a data point either belongs to that cluster or it doesn't. On the other hand, clustering methods such as Gaussian Mixture Models (GMM) have soft boundaries, where data points can belong to multiple cluster at the same time but with different degrees of belief. e.g. a data point can have a 60% of belonging to cluster 1, 40% of belonging to cluster 2.
# 
# Apart from using it in the context of clustering, one other thing that GMM can be useful for is outlier detection: Due to the fact that we can compute the likelihood of each point being in each cluster, the points with a "relatively" low likelihood (where "relatively" is a threshold that we just determine ourselves) can be labeled as outliers.

# This is the exact situation we're in when doing GMM. We have a bunch of data points, we suspect that they came from  K  different guassians, but we have no clue which data points came from which guassian. To solve this problem, we use the EM algorithm. The way it works is that it will start by placing guassians randomly (generate random mean and variance for the guassian). Then it will iterate over these two steps until it converges.
# 
# E step : With the current means and variances, it's going to figure out the probability of each data point  xi  coming from each guassian.
# M step  : Once it computed these probability assignments it will use these numbers to re-estimate the guassians' mean and variance to better fit the data points.

# That could be a problem for datasets with large number of dimensions (e.g. text data), because with the number of parameters growing roughly as the square of the dimension, it may quickly become impossible to find a sufficient amount of data to make good inferences. One common way to avoid this problem is to fix the covariance matrix of each component to be diagonal (off-diagonal value will be 0 and will not be updated). To achieve this, we can change the covariance_type parameter in scikit-learn's GMM to diag.

# In[8]:


X2 = pid2                  # X denotes the input functions and here class defines whether the person is ill or not
print(X2)
y2 = pid['class']                  #y denotes the output functions
print(y2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, 
                                                    train_size=0.7, 
                                                    random_state=42,
                                                    stratify=y2)

print(X2_train)
print(X2_test)
print(y2_train)
print(y2_test)


# In[9]:


import numpy as np
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=2,covariance_type='full',max_iter=1000,n_init=2,init_params='kmeans', random_state=42).fit(pid2)
clf=gm.fit(X2_train, y2_train)
y_pred=gm.predict(X2_test)       #NB classifier assumes that all the features are indipendent to each other
print(y_pred)


# In[10]:


print(clf.bic(pid2))
print(clf.aic(pid2))


# In[11]:


n_estimators=np.arange(1,8)
clfs=[GaussianMixture(n ).fit(pid2) for n in n_estimators]
bics=[clf.bic(pid2) for clf in clfs]
aics=[clf.aic(pid2) for clf in clfs]
print(clfs)
print(bics)
print(aics)
plt.plot(n_estimators, bics, label="BIC")    #low value of both aic and bic is preffered
plt.plot(n_estimators, aics, label="AIC")
plt.legend();


# In[12]:


import numpy as np
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=6,covariance_type='full',max_iter=1000,n_init=2,init_params='kmeans', random_state=42).fit(pid2)
clf=gm.fit(X2_train, y2_train)
y_pred=gm.predict(X2_test)       #NB classifier assumes that all the features are indipendent to each other
print(y_pred)

print('\n\n')
print(gm.means_)
print('\n\n')
print(gm.score(pid2, y=None))
print('\n\n')
probs = gm.predict_proba(pid2)
print(probs) 


# In[13]:


accuracy=accuracy_score(y2_test, y_pred)*100
print('accuracy = ',accuracy,'%')
print('\n')
matrix=confusion_matrix(y2_test,y_pred)
print('Confusion Matrix = ',matrix)


# In[14]:


import numpy as np
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=4,covariance_type='full',max_iter=1000,n_init=1,init_params='kmeans', random_state=42).fit(pid2)
print(gm.means_)
print('\n\n')
print(gm.score(pid2, y=None))
print('\n\n')
probs = gm.predict_proba(pid2)
print(probs)          #[:8].round(3))


# In[ ]:




