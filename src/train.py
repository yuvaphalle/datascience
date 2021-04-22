#!/usr/bin/env python
# coding: utf-8

# ### Installing

# In[1]:


# !pip install scikit-learn==0.24.0
# !curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


# !pip install auto-sklearn


# In[4]:


# pip install dask distributed


# ### Importing

# In[5]:


from google.colab import drive
import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import logging
import datetime
from joblib import dump


# ### Google Drive connection

# In[6]:


mount_path = '/content/drive'
drive.mount(mount_path, force_remount=True)


# In[7]:


data_path = "/content/drive/MyDrive/ml data/"
model_path = "/content/drive/My Drive/Introduction2DataScience/w2d2/models/"


# In[8]:


timesstr = str(datetime.datetime.now()).replace(' ', '_')
logging.basicConfig(filename=f"{model_path}explog_{timesstr}.log", level=logging.INFO)


# In[9]:


#set_config(display='diagram')


# In[ ]:


wine = pd.read_csv(f'{data_path}winequality-red.csv', sep=';')


# In[ ]:


test_size = 0.2
random_state = 0
train, test = train_test_split(wine, test_size=test_size, random_state=random_state)
train.to_csv(f'{data_path}winequality-red-train.csv', index=False, sep=';')
train = train.copy()
test.to_csv(f'{data_path}winequality-red-test.csv', index=False, sep=';')
test = test.copy()


# In[ ]:


logging.info(f'train test split with test_size={test_size} and random state={random_state}')


# <a id='P2' name="P2"></a>
# ## [Modelling](#P0)

# In[ ]:


X_train, y_train = train.iloc[:,:-1], train.iloc[:,-1]


# ### Pipeline Definition

# In[ ]:


from sklearn.linear_model import LinearRegression
total_time = 600
per_run_time_limit = 30

automl = LinearRegression()


# ### Model Training

# In[ ]:


automl.fit(X_train, y_train)


# In[ ]:


logging.info(f'Ran autosklearn regressor for a total time of {total_time} seconds, with a maximum of {per_run_time_limit} seconds per model run')


# In[ ]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[ ]:


logging.info(f'Saved regressor model at {model_path}model{timesstr}.pkl ')


# In[ ]:


# logging.info(f'autosklearn model statistics:')
# logging.info(automl.sprint_statistics())


# ### Model Evaluation

# In[ ]:


X_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]

y_pred = automl.predict(X_test)
y_pred


# _Mean Squared Error:_

# In[ ]:


mse = mean_squared_error(y_test, y_pred)
mse


# _R^2 score:_

# In[ ]:


R_squared = automl.score(X_test, y_test)
R_squared


# In[ ]:


logging.info(f"Mean Squared Error is {mse}, \n R2 score is {R_squared}")

