#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction

# We are using Kaggle’s Red Wine Quality dataset or the dataset was already mentioned in the assignment email to build various classification models to predict whether a particular red wine is “good quality” or not. Each wine in this dataset is given a “quality” score between 0 and 10. For the purpose of this project, I converted the output to a binary output where each wine is either “good quality” (a score of 7 or higher) or not (a score below 7). The quality of a wine is determined by 11 input variables:
# 1. Fixed acidity
# 2. Volatile acidity
# 3. Citric acid
# 4. Residual sugar
# 5. Chlorides
# 6. Free sulfur dioxide
# 7. Total sulfur dioxide
# 8. Density
# 9. pH
# 10. Sulfates
# 11. Alcohol

# # Objectives
# 
# The objectives of this project are as follows:
# 
# => To determine which features are the most indicative of a good quality wine
# 
# => To experiment with different machine learning algorithm to see which yields the highest accuracy

# # Dataset Description
# 
# You can see that several features will be used to classify the quality of wine, many of them are chemical, so we need to have a basic understanding of such chemicals
# 
# * volatile acidity :   Volatile acidity is the gaseous acids present in wine.
# * fixed acidity :   Primary fixed acids found in wine are tartaric, succinic, citric, and malic
# * residual sugar :   Amount of sugar left after fermentation.
# * citric acid :    It is weak organic acid, found in citrus fruits naturally.
# * chlorides :   Amount of salt present in wine.
# * free sulfur dioxide :   So2 is used for prevention of wine by oxidation and microbial spoilage.
# * total sulfur dioxide 
# * pH :   In wine pH is used for checking acidity
# * density : Depends on the alcohol and sugar content. Better wines usually have lower densities.
# * sulphates :    Added sulfites preserve freshness and protect wine from oxidation, and bacteria.
# * alcohol :   Percent of alcohol present in wine.

# ## Importing Libraries

# In[1]:


# Import required libraris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# In[2]:


# Loading wine quality dataset
df=pd.read_csv('Wine-Quality-Dataset.csv')
print('(Rows, Columns):',str(df.shape))


# In[3]:


# Check starting 5 rows
df.head()


# ## Exploratory Data Analysis

# In[4]:


print(df.isnull().sum())


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


# Checking correlation with quality
df.corr()['quality'].plot(kind='barh')


# In[8]:


# Viualization of Quality feature
fig=px.histogram(df,x='quality')
fig.show()


# In[9]:


# All posible hist plots
df.hist(bins=25,figsize=(10,10))
plt.show()


# In[10]:


plt.figure(figsize=[10,6])

plt.bar(df['quality'],df['alcohol'],color='red')
plt.xlabel('quality')
plt.ylabel('alcohol')


# In[11]:


# Correlation matrix
corr=df.corr()
plt.subplots(figsize=(15,10))

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,annot=True,cmap=sns.diverging_palette(22,0,20,as_cmap=True))


# In[12]:


# Create Classification version of target variable
df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]


# In[13]:


# Seperate Feature variable and target variable
X=df.drop(['quality','goodquality'],axis=1)
y=df['goodquality']


# In[14]:


df['goodquality'].value_counts()


# In[15]:


sns.pairplot(df, hue='goodquality')


# By looking above distribution of data we select following algorithms:
# 
# * Decision Tree
# * Random Forest
# * AdaBoost
# * Gradiand Boosting
# * XGBoost

# ## Standardization Features

# In[16]:


# Standardizing the feature variables
from sklearn.preprocessing import StandardScaler

X_feature=X
X=StandardScaler().fit_transform(X)


# ## Spliting and applying algorithms

# In[17]:


# Spliting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=.25, random_state=0)


# In[18]:


from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

tree_model=DecisionTreeClassifier(random_state=1)
tree_model.fit(X_train, y_train)
tree_pred=tree_model.predict(X_test)

print(classification_report(y_test, tree_pred))


# In[19]:


from sklearn.ensemble import RandomForestClassifier

RF_model=RandomForestClassifier(random_state=1)
RF_model.fit(X_train, y_train)
RF_pred=RF_model.predict(X_test)

print(classification_report(y_test, RF_pred))


# In[20]:


from sklearn.ensemble import AdaBoostClassifier

adbc_model=AdaBoostClassifier(random_state=1)
adbc_model.fit(X_train, y_train)
adbc_pred=adbc_model.predict(X_test)

print(classification_report(y_test, adbc_pred))


# In[21]:


from sklearn.ensemble import GradientBoostingClassifier

gbc_model=GradientBoostingClassifier(random_state=1)
gbc_model.fit(X_train, y_train)
gbc_pred=gbc_model.predict(X_test)

print(classification_report(y_test, gbc_pred))


# In[22]:


from xgboost import XGBClassifier

XG_model=XGBClassifier(random_state=1)
XG_model.fit(X_train, y_train)
XG_pred=XG_model.predict(X_test)

print(classification_report(y_test, XG_pred))


# ## Finding Important Featues

# In[23]:


feature_importance=pd.Series(RF_model.feature_importances_,index=X_feature.columns)
feature_importance.nlargest(25).plot(kind='barh',figsize=(10,5))


# In[24]:


feature_importance=pd.Series(XG_model.feature_importances_,index=X_feature.columns)
feature_importance.nlargest(25).plot(kind='barh',figsize=(10,5))


# In[25]:


import pickle

# Serialize and save the model to a file
with open('my_model.pkl', 'wb') as file:
    pickle.dump(XG_model, file)


# #Load the serialized model from file
# 
# with open('my_model.pkl', 'rb') as file:
#     
#     loaded_model = pickle.load(file)

# In[26]:


good_quality = df[df['quality']>=7]


# In[28]:


good_quality.head()

