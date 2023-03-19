#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/Win11/Documents/Industry_Projects/The_Spark_Foundation/Regression.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


df.describe()


# In[9]:


df.value_counts().head(10)


# In[10]:


plt.title('Regression Plot')
plt.scatter(x='Hours', y='Scores',s=10, c='r',marker='*',linewidth=3,data=df)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[11]:


sns.regplot(x='Hours',y='Scores',label= 'Regplot',data=df)


# In[12]:


df.corr()


# In[13]:


sns.heatmap(df.corr())


# In[14]:


pip install statsmodels


# In[15]:


df.head()


# In[16]:


import statsmodels.api as sm
df = sm.add_constant(df['Hours'])
df.head(5)


# In[17]:


x= df.iloc[:, :-1].values  
y= df.iloc[:, 1].values


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state =0 )


# In[23]:



from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
slope = regressor.coef_
intercept = regressor.intercept_
print(intercept)
print(slope)


# In[20]:


df.head()


# In[24]:


# prediction of the score by studying 9.5 hours

Result = 5.24*9.5 + 1.0


# In[26]:


print(Result)

