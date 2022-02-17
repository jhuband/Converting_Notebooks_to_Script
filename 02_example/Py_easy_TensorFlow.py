#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Example using Keras
# 
# ## Load the keras packages

# In[1]:


from keras.models import Sequential 
from keras.layers import Dense, Dropout 
from keras.utils import to_categorical 
from keras.optimizers import SGD


# In[2]:


import numpy as np
data_file = 'cancer_data.csv'
target_file = 'cancer_target.csv'
cancer_data=np.loadtxt(data_file,dtype=float,delimiter=',')
cancer_target=np.loadtxt(target_file, dtype=float, delimiter=',')


# In[3]:


from sklearn import model_selection
test_size = 0.20 
seed = 7 
data = model_selection.train_test_split(cancer_data,  cancer_target, test_size=test_size, random_state=seed)

train_data = data[0]
test_data = data[1]
train_target = data[2]
test_target = data[3]


# In[4]:


from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
# Fit the scaler only to the training data 
scaler.fit(train_data)

# Now apply the transformations to the training and test data: 
x_train = scaler.transform(train_data) 
x_test = scaler.transform(test_data)

# Convert the classes to ‘one-hot’ vector 
y_train = to_categorical(train_target, num_classes=2) 
y_test = to_categorical(test_target, num_classes=2)


# In[5]:


model = Sequential() 
# In the first layer, you must specify the expected 
#  input data shape
# here, 30-dimensional vectors. 
model.add(Dense(30, activation='relu', input_dim=30)) 
model.add(Dense(60, activation='relu')) 
model.add(Dense(2, activation='softmax')) 
print(model.summary())


# In[6]:


model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])


# In[7]:


b_size = int(.8*x_train.shape[0])

model.fit(x_train, y_train, epochs=300, batch_size=b_size)


# In[8]:


#predictions = model.predict_classes(x_test)
predictions = np.argmax(model.predict(x_test), axis=-1)


# In[9]:


score = model.evaluate(x_test, y_test, batch_size=b_size) 
print('\nAccuracy:  %.3f' % score[1])
from sklearn.metrics import confusion_matrix 
print(confusion_matrix(test_target, predictions))

