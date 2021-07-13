#!/usr/bin/env python
# coding: utf-8

# # Assignment_7
# 
# Submitted By - Saksham Chauhan

# In[18]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import patches


# In[19]:


df= pd.read_csv("pima-indians-diabetes.csv")
df.head()


# In[20]:


C=df.drop(["class"],axis=1)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X=sc.fit_transform(C)

Y=df["class"]


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,
                                test_size=0.30,random_state=550)


# # Using KNN

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
acc=[]
k=[]
confuse=[]
for i in range(1,22,+2):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc.append(accuracy_score(y_test,y_pred))
    confuse.append(confusion_matrix(y_test,y_pred))
    k.append(i)


# In[23]:


data=pd.DataFrame({"VALUE OF K":k,"ACCURCY":acc, "Conusion_matrix":confuse})

print(data)


# # Using Bayes Classifier

# In[5]:


C=df.drop(["class"],axis=1)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X=sc.fit_transform(C)

Y=df["class"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,
                                test_size=0.30,random_state=550)


# In[6]:


num_classes=len(np.unique(y_train))
print(num_classes)


# # Q_component = 1

# In[15]:


from sklearn import mixture
from sklearn.metrics import accuracy_score,confusion_matrix
classifier_0 = mixture.GaussianMixture(n_components=1, covariance_type='full')
classifier_0.means_ = np.array([x_train[y_train == i].mean(axis=0) for i in range(num_classes)])
classifier_0.fit(x_train)
y_train_pred_0 = classifier_0.predict(x_train)
accuracy_training_0 = np.mean(y_train_pred_0.ravel() == y_train.ravel()) * 100
print('Accuracy on the training data =', accuracy_training_0)

y_test_pred_0 = classifier_0.predict(x_test)
accuracy_testing_0 = np.mean(y_test_pred_0.ravel()==y_test.ravel())* 100
print('Accuracy on testing data =', accuracy_testing_0)

print(confusion_matrix(y_train,y_train_pred_0))
print(confusion_matrix(y_test,y_test_pred_0))


# # Q_component = 2

# In[16]:


from sklearn import mixture

classifier = mixture.GaussianMixture(n_components=num_classes, covariance_type='full')
classifier.means_ = np.array([x_train[y_train == i].mean(axis=0) for i in range(num_classes)])
classifier.fit(x_train)
y_train_pred = classifier.predict(x_train)
accuracy_training = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
print('Accuracy on the training data =', accuracy_training)

y_test_pred = classifier.predict(x_test)
accuracy_testing = np.mean(y_test_pred.ravel()==y_test.ravel())* 100
print('Accuracy on testing data =', accuracy_testing)

print(confusion_matrix(y_train,y_train_pred))
print(confusion_matrix(y_test,y_test_pred))


# # Q_component = 4

# In[17]:


from sklearn import mixture
classifier_1 = mixture.GaussianMixture(n_components=4, covariance_type='full')
classifier_1.fit(x_train)
y_train_pred_1 = classifier_1.predict(x_train)
accuracy_training_1 = np.mean(y_train_pred_1.ravel() == y_train.ravel()) * 100
print('Accuracy on the training data =', accuracy_training_1)

y_test_pred_1 = classifier_1.predict(x_test)
accuracy_testing_1 = np.mean(y_test_pred_1.ravel()==y_test.ravel())* 100
print('Accuracy on testing data =', accuracy_testing_1)

print(confusion_matrix(y_train,y_train_pred_1))
print(confusion_matrix(y_test,y_test_pred_1))


# # Q_component = 8

# In[14]:


from sklearn import mixture
classifier_2 = mixture.GaussianMixture(n_components=8, covariance_type='full')
classifier_2.fit(x_train)
y_train_pred_2 = classifier_2.predict(x_train)
accuracy_training_2 = np.mean(y_train_pred_2.ravel() == y_train.ravel()) * 100
print('Accuracy on the training data =', accuracy_training_2)

y_test_pred_2 = classifier_2.predict(x_test)
accuracy_testing_2 = np.mean(y_test_pred_2.ravel()==y_test.ravel())* 100
print('Accuracy on testing data =', accuracy_testing_2)

print(confusion_matrix(y_train,y_train_pred_2))
print(confusion_matrix(y_test,y_test_pred_2))


# # Q_component = 16

# In[13]:


from sklearn import mixture
classifier_3 = mixture.GaussianMixture(n_components=16, covariance_type='full')
classifier_3.fit(x_train)
y_train_pred_3 = classifier_3.predict(x_train)
accuracy_training_3 = np.mean(y_train_pred_3.ravel() == y_train.ravel()) * 100
print('Accuracy on the training data =', accuracy_training_3)

y_test_pred_3 = classifier_3.predict(x_test)
accuracy_testing_3 = np.mean(y_test_pred_3.ravel()==y_test.ravel())* 100
print('Accuracy on testing data =', accuracy_testing_3)

print(confusion_matrix(y_train,y_train_pred_3))
print(confusion_matrix(y_test,y_test_pred_3))

