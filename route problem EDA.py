#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[2]:


df = pd.read_csv("route_data.csv")


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.columns


# In[7]:


df.shape


# In[8]:


df.dtypes


# In[9]:


df.boxplot(['declared_quantity','days_in_transit'],figsize=(10,11))


# In[10]:


#dropping columns having null values and also no useful
df.drop(['item'], axis=1, inplace=True)
df.drop(['importer_id'], axis=1, inplace=True)
df.drop(['exporter_id'], axis=1, inplace=True)
df.drop(['mode_of_transport'], axis=1, inplace=True)


# In[11]:


df.head()


# In[12]:


#this tells about frequency distribution of categories within the feature
df['route'].value_counts()


# In[13]:


print(df['route'].value_counts().count())


# In[14]:


catge_count = df['route'].value_counts()
sns.set(style="darkgrid")
sns.barplot(catge_count.index, catge_count.values, alpha=0.9)
plt.title('Frequency Distribution of Catgeroicals')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Carrier', fontsize=12)
plt.show()


# In[15]:


dummies = pd.get_dummies(df['route'], prefix='route')
df = pd.concat([df, dummies], axis=1)


# In[16]:


print(dummies.head())


# In[17]:


df.drop(['route'], axis=1, inplace=True)


# In[18]:


dummies = pd.get_dummies(df['country_of_origin'], prefix='origin')
df = pd.concat([df, dummies], axis=1)


# In[19]:


dummies.head()


# In[20]:


df.drop(['country_of_origin'], axis=1, inplace=True)


# In[21]:


#dropping below columns because all date fileds are same thing.
#df.drop(['date_of_arrival'], axis=1, inplace=True)
#df.drop('date_of_departure', axis=1, inplace=True)


# In[22]:


df['weight_diff'] = ((df['actual_weight']-df['declared_weight']) / df['actual_weight']).round(3)*100


# In[23]:


df['date_of_arrival'] = pd.to_datetime(df['date_of_arrival'])
df['date_of_departure'] = pd.to_datetime(df['date_of_departure'])
df['days_diff'] = df['date_of_arrival'] - df['date_of_departure']
df['days_diff'] = df['days_diff'] / np.timedelta64(1,'D')


# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
x = df['declared_weight'].values 
y= df['actual_weight'].values 
reg = LinearRegression()
x_reshaped = x.reshape((-1,1)) 
regression = reg.fit(x_reshaped,y)
prev = reg.predict(x_reshaped)
print('Y = {}X {}'.format(reg.coef_,reg.intercept_))

R_2 = r2_score(y, prev) 
print("Coefficient of determination (R2):", R_2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




