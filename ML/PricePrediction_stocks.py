#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df=pd.read_csv("31-12-2021-TO-30-12-2022TATASTEELALLN.csv")
print('Author:- Barun Prakash ,Date:-24/06/23, Price prediction')
df


# In[3]:


df.shape


# In[10]:


df.columns


# In[4]:


df[df.Average_Price >100]
#df1['Average_Price'] =df['Average_Price']>800


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("Deliverable_Qty")
plt.ylabel("Average_Price")
plt.scatter(df.Deliverable_Qty ,df.Average_Price,color='red',marker="+")


# In[12]:


LinearReg = linear_model.LinearRegression()
LinearReg.fit(df[['Deliverable_Qty']],df.Average_Price)


# In[13]:


LinearReg.predict([[10253413]])


# In[11]:


reg.coef_


# In[ ]:


reg.inter

