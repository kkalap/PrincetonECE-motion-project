#!/usr/bin/env python
# coding: utf-8

# In[47]:


# load dataset 
import matplotlib. pyplot as plt
import pandas as pd

# load dataset into Pandas DataFrame
df = pd.read_csv('/Users/katharinekalap/Desktop/hossein proj/data/training/trainingdata.csv', encoding='latin1',delimiter=',', header=None, skiprows=1, names=['orientation(sys)','gyroscopex','gyroscopey','gyroscopez','accelerationx','accelerationy','accelerationz','magnonmeterx','magnometery','magnometerz', 'movement'])


# In[54]:


# standardise data 
from sklearn.preprocessing import StandardScaler
features =['orientation(sys)','gyroscopex','gyroscopey','gyroscopez','accelerationx','accelerationy','accelerationz','magnonmeterx','magnometery','magnometerz']

# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['movement']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)


# In[56]:


# convert into 2 dimensions 
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['movement']]], axis = 1)


# In[59]:


# visualise 
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['starjumps', 'pressups']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['movement'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[ ]:





# In[ ]:




