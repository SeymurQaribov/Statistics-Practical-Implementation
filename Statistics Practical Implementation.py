#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import statistics
import seaborn as sns
import math
import matplotlib.pyplot as plt
import scipy.stats as stat
import pylab


# In[3]:


age =[23,24,45,32,12,56,34,23,87,90,46,24,34,46,123,100,23,45,67,89,98]


# In[13]:


np.mean(age),3


# In[6]:


np.median(age)


# In[10]:


statistics.median(age)


# In[11]:


statistics.mean(age)


# In[14]:


statistics.mode(age)


# In[15]:


sns.boxplot(age)


# In[22]:


q1,q3 = np.percentile(age,[25,75])


# In[23]:


q1,q3


# In[44]:


#to check outlier

IQR = q3 - q1
Lower_fence = q1 - 1.5*IQR
Higer_fence = q3 + 1.5*IQR
print(Lower_fence,Higer_fence)


# In[45]:


statistics.variance(age)


# In[46]:


statistics.pvariance(age)


# In[47]:


math.sqrt(statistics.pvariance(age))


# In[48]:


np.var(age,axis= 0)


# In[49]:


#Population viraince

def variance(data):
    n = len(age)
    mean = sum(data)/n
    deviation  = [(x-mean)**2 for x in data]
    variance = sum(deviation)/n
    return variance
print(variance(age))


# In[50]:


#Sample viriance

def variance(data):
    n = len(age)
    mean = sum(data)/n
    deviation  = [(x-mean)**2 for x in data]
    variance = sum(deviation)/n-1
    return variance
print(variance(age))


# In[51]:


def variance(data, dof = 0):
    n = len(age)
    mean = sum(data)/n
    deviation = [(x-mean)**2  for x in data]
    variance = sum(deviation)/(n-dof)
    return variance
print(variance(age))



# In[52]:


def variance(data, dof= 1):
    n = len(age)
    mean = sum(data)/n
    deviation = [(x-mean)**2 for x in data]
    varinace = sum(deviation)/(n-dof)
    return variance
print(variance(age))


# In[53]:


sns.histplot(age, kde = True)


# In[54]:


s = np.random.normal(0.5,0.2,1000)


# In[55]:


s


# In[57]:


sns.histplot(s,kde = True)


# In[61]:


mu,sigma = 3,1
s1 = np.random.lognormal(mu,sigma,1000)


# In[62]:


s1


# In[63]:


sns.histplot(s1, kde = True)


# In[64]:


sns.histplot(np.log(s1),kde = True)


# In[70]:


def data_plot(sample):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    sns.histplot(sample)
    plt.subplot(1,2,2)
    stat.probplot(sample, dist = 'norm',plot = pylab)
    plt.show()
    
     
    
  

    


# In[71]:


s2 = np.random.normal(0.5,0.2,1000)
data_plot(s2)


# In[ ]:





# In[ ]:




