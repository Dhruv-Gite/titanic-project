#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[79]:


titanic = pd.read_csv('Titanic-Dataset.csv')


# In[80]:


titanic


# In[81]:


titanic.info()


# In[82]:


titanic.isnull().sum()


# In[83]:


titanic.describe()


# In[84]:


titanic.duplicated().sum()


# In[85]:


titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())


# In[86]:


titanic


# In[87]:


titanic.drop(['Cabin','Ticket','PassengerId','Name'],axis=1,inplace=True)


# In[88]:


titanic


# In[89]:


titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])


# In[90]:


titanic


# In[91]:


titanic.isnull().sum()


# In[92]:


titanic['Survived'].value_counts().plot(kind='bar')


# In[93]:


b = titanic.groupby('Sex')['Survived'].sum().plot(kind='bar')


# In[94]:


titanic['Age'].value_counts()


# In[95]:


#encoding


# In[96]:


titanic_encoded = pd.get_dummies(
    titanic,
    columns=['Sex','Embarked'],
    drop_first=True,
)


# In[97]:


titanic_encoded


# In[98]:


# Feature Engineering


# In[99]:


titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1


# In[100]:


titanic['IsAlone'] = (titanic['FamilySize'] == 1).astype(int)


# In[101]:


titanic


# In[102]:


titanic = pd.get_dummies(
    titanic,
    columns=['Sex','Embarked'],
    drop_first=True
)


# In[103]:


titanic


# In[104]:


x = titanic.drop('Survived',axis=1)


# In[105]:


y = titanic['Survived']


# In[106]:


x


# In[107]:


y


# In[108]:


x_train, x_test, y_train, y_test=train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42
)



# In[109]:


logistic_model = LogisticRegression()


# In[110]:


logistic_model.fit(x_train,y_train)


# In[111]:


prediction = logistic_model.predict(x_test)
if prediction[0] == 1:
    print('Predicted: Survived')
else:
    print('Predicted: Did Not Survived')


# In[112]:


joblib.dump(logistic_model,'lregration.pkl')


# In[113]:


# Model 2 - KNN


# In[114]:


knn_model = KNeighborsClassifier(
    n_neighbors=5
)


# In[115]:


knn_model.fit(
    x_train, y_train
)


# In[116]:


knn_pred = knn_model.predict(x_test)
if knn_pred[0] == 1:
    print('Predicted: Survived')
else:
    print('Predicted: Did Not Survived')


# In[121]:


joblib.dump(knn_model,'knn.pkl')


# In[122]:


# model 3 - SVM


# In[123]:


svm_model = SVC(
    kernel = 'rbf',
    probability = True,
    random_state=42
)


# In[124]:


svm_model.fit(
    x_train, y_train
)


# In[125]:


svm_pred = svm_model.predict(x_test)
if svm_pred[0] == 1:
    print('Predicted: Survived')
else:
    print('Predicted: Did Not Survived')


# In[126]:


joblib.dump(svm_model,'svm.pkl')


# In[ ]:




