#!/usr/bin/env python
# coding: utf-8

# ##  Random Forest Example

# #### Step 1.  Read data from URL
# 

# In[1]:


import pandas as pd

csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
raw_data = pd.read_csv(csv_url, sep=";")
raw_data.head()


# ### Step 2. Split data into features (X) and labels (y)

# In[2]:


X = raw_data.loc[:, raw_data.columns != 'quality']
X.head()
y = raw_data.loc[:, raw_data.columns == 'quality']
y.head()
print(X.shape, y.shape)


# ### Step 3.  Convert labels to expected format (e.g., 1-D array)

# In[3]:


import numpy as np
y_np = y.values.flatten()

print(y_np)
type(y_np)


# ### Step 4.  Set up classifier & fit the data

# In[4]:



from sklearn.ensemble import RandomForestClassifier


# In[5]:


model = RandomForestClassifier()
model.fit(X,y_np)


# ### Step 5.  Compare the prediction to a random row

# In[6]:


import random
random_row = random.randint(0, X.shape[0])
random_row


# In[7]:



row = [X.values[random_row,:]]

y_hat = model.predict(row)
print('Prediction:  %d' % y_hat)
print('Actual:      %d' % y.values[random_row])


# ### Step 6.  View Feature importance

# In[8]:


importances = model.feature_importances_
feature_names = [f"feature {i}" for i in range(X.shape[1])]
feature_importances = pd.Series(importances, index=feature_names)


# In[9]:


import matplotlib.pyplot as plt
indices = np.argsort(importances)


fig, ax = plt.subplots()
ax.barh(range(len(importances)), importances[indices])
ax.set_yticks(range(len(importances)))
_ = ax.set_yticklabels(np.array(X.columns)[indices])

## Save plot to a file
plt.savefig('Wine_Features.png')

