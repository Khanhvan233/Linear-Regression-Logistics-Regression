#!/usr/bin/env python
# coding: utf-8

# # Data Information
# 

#  Citation Request:
#   This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
#   Please include this citation if you plan to use this database:
# 
#   P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
#   Modeling wine preferences by data mining from physicochemical properties.
#   In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
# 
#   Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
#                 [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
#                 [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib
# 
# 1. Title: Wine Quality 
# 
# 2. Sources
#    Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009
#    
# 3. Past Usage:
# 
#   P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
#   Modeling wine preferences by data mining from physicochemical properties.
#   In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
# 
#   In the above reference, two datasets were created, using red and white wine samples.
#   The inputs include objective tests (e.g. PH values) and the output is based on sensory data
#   (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality 
#   between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model
#   these datasets under a regression approach. The support vector machine model achieved the
#   best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T),
#   etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity
#   analysis procedure).
#  
# 4. Relevant Information:
# 
#    The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
#    For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
#    Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables 
#    are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
#    These datasets can be viewed as classification or regression tasks.
#    The classes are ordered and not balanced (e.g. there are munch more normal wines than
#    excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent
#    or poor wines. Also, we are not sure if all input variables are relevant. So
#    it could be interesting to test feature selection methods. 
# 
# 5. Number of Instances: red wine - 1599; white wine - 4898. 
# 
# 6. Number of Attributes: 11 + output attribute
#   
#    Note: several of the attributes may be correlated, thus it makes sense to apply some sort of
#    feature selection.
# 
# 7. Attribute information:
# 
#    For more information, read [Cortez et al., 2009].
# 
#    Input variables (based on physicochemical tests):
#    1 - fixed acidity
#    2 - volatile acidity
#    3 - citric acid
#    4 - residual sugar
#    5 - chlorides
#    6 - free sulfur dioxide
#    7 - total sulfur dioxide
#    8 - density
#    9 - pH
#    10 - sulphates
#    11 - alcohol
#    Output variable (based on sensory data): 
#    12 - quality (score between 0 and 10)
# 
# 8. Missing Attribute Values: None

# # Add Module
# 

# In[96]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# # Read Data From CSV

# In[9]:


#Read data from csv
wine_dataset = pd.read_csv('winequality-red.csv',sep=';')


# # Preprocessing Data
# 

# In[10]:


#Shape of the data
wine_dataset.shape


# In[17]:


wine_dataset.isnull().sum()


# In[11]:


wine_dataset.head()


# In[12]:


wine_dataset.isnull().sum()


# In[13]:


wine_dataset.describe()


# In[34]:


wine_dataset['quality'].value_counts()


# In[44]:


style.use('ggplot')
sns.countplot(x='quality' ,data=wine_dataset)


# In[45]:


wine_dataset.hist(bins=100, figsize=(10,12))
plt.show()


# In[47]:


#check which element effect the quality the most
s=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
for i in s:
    plot = plt.figure(figsize=(4,4))
    sns.barplot(x='quality', y=i, data=wine_dataset)
 


# In[24]:


corr = wine_dataset.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# # Split Processing
# 
# 

# In[52]:


#Quality >7 mean good wine and = 1
#Quality <7 mean not good wine and =0
wine_dataset['quality'] = wine_dataset.quality.apply(lambda x:1 if x>=7 else 0)


# In[53]:


wine_dataset['quality'].value_counts()


# In[55]:


X = wine_dataset.drop('quality', axis=1)
y = wine_dataset['quality']


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[59]:


print("X_train ", X_train.shape)
print("y_train ", y_train.shape)
print("X_test ", X_test.shape)
print("y_test ", y_test.shape)


# # Training Module
# 

# ## Multiple Regression Linear

# In[101]:


# Create logistic regression
linear_reg = LinearRegression()

# Training variable
linear_reg.fit(X_train, y_train)

# Prediction
linear_reg_pred = linear_reg.predict(X_test)

#Mean squared error
mse = mean_squared_error(y_test, linear_reg_pred)
print("Mean Squared Error is: {:.2f}".format(mse))

# Calculate R-squared
r2 = r2_score(y_test, linear_reg_pred)
print("R-Squared: {:.2f}".format(r2))

# Calculate mean absolute error
mae = mean_absolute_error(y_test, linear_reg_pred)
print("Mean Absolute Error: {:.2f}".format(mae))


# In[100]:


metrics = ['R-squared', 'Mean Squared Error', 'Mean Absolute Error']
values = [r2, mse, mae]

plt.bar(metrics, values)
plt.ylabel('Value')
plt.title('Evaluation Metrics')
plt.show()


# ## Logistics Regression

# In[83]:


# Create logistic regression
log_reg = LogisticRegression()

# Training variable
log_reg.fit(X_train, y_train)

# Prediction
log_reg_pred = log_reg.predict(X_test)

# Accurance testing
log_reg_acc = accuracy_score(log_reg_pred, y_test)
print("Test accuracy is: {:.2f}%".format(log_reg_acc*100))


# In[87]:


#Confusion matrix
#TN (True Negative), FN (False Negative), TP (True Positive) and FP (False Positive) 
style.use('ggplot')
cm = confusion_matrix(y_test, log_reg_pred, labels=log_reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=log_reg.classes_)
disp.plot()
print("TN: ", cm[0][0])
print("FN: ", cm[1][0])
print("TP: ", cm[1][1])
print("FP: ", cm[0][1])


# In[ ]:




