#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

# libraby for linear algebra
import numpy as np 

# library for data processing
import pandas as pd 

# library for data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

# Libraries for different libraries
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# In[2]:


# Read the csv file using 'read_csv'
test_df = pd.read_csv('test_Yix80N0.csv')
train_df = pd.read_csv('train_8wry4cB.csv')


# In[3]:


# Top 5 records of train dataframe
train_df.head()


# In[4]:


# Check the column-wise info of the train dataframe
train_df.info()


# In[5]:


# Get a summary of the train dataframe using 'describe()'
train_df.describe()


# In[6]:


# Check the number of rows and columns in the dataframes
print(train_df.shape)
print(test_df.shape)


# In[7]:


# Top 5 records of test dataframe
test_df.head()


# In[8]:


# Get the column-wise Null count
train_df.isnull().sum()


# ## Data Prepration

# In[9]:


# Extract the new column mnth_yr and hour to see the month of shopping and 
train_df['startTime'] = pd.to_datetime(train_df['startTime'])
test_df['startTime'] = pd.to_datetime(test_df['startTime'])
train_df['month'] = train_df['startTime'].dt.month
test_df['month'] = test_df['startTime'].dt.month
train_df['week'] = train_df['startTime'].dt.week
test_df['week'] = test_df['startTime'].dt.week
train_df['hour'] = train_df['startTime'].dt.hour
test_df['hour'] = test_df['startTime'].dt.hour


# In[10]:


train_df["ProductList"]= train_df["ProductList"].str.split(";", n = 1, expand = True)
test_df["ProductList"]= test_df["ProductList"].str.split(";", n = 1, expand = True)


# In[11]:


# new data frame with split value columns 
train_product_list = train_df["ProductList"].str.split("/", expand = True)

# making separate first name column from new data frame 
train_df['Prod_Category_1']= train_product_list[0]
train_df['Prod_Category_2']= train_product_list[1]
train_df['Prod_Category_3']= train_product_list[2]
train_df['Prod_Category_4']= train_product_list[3]
train_df.head()


# In[12]:


# new data frame with split value columns 
test_product_list = test_df["ProductList"].str.split("/", expand = True)

# making separate first name column from new data frame 
test_df['Prod_Category_1']= test_product_list[0]
test_df['Prod_Category_2']= test_product_list[1]
test_df['Prod_Category_3']= test_product_list[2]
test_df['Prod_Category_4']= test_product_list[3]
test_df.head()


# In[13]:


# Drop the unnecessary columns from the train and test dataframe
train_df.drop(['ProductList','session_id','startTime','endTime'], axis=1, inplace=True)
test_df.drop(['ProductList','session_id','startTime','endTime'], axis=1, inplace=True)


# ## Data Visualization

# In[14]:


# find out the distinct hotel id’s 
train_df['gender'].astype('category').value_counts()


# In[15]:


# Countplot to see the distribution of genders in train dataframe
sns.countplot(train_df['gender'])
plt.xlabel("Gender (0 = female, 1= male)")
plt.show()


# In[16]:


plt.figure(figsize=(15,6))
sns.countplot(train_df['Prod_Category_1'])
plt.xlabel("Different Products in Category_1")
plt.show()


# In[17]:


pd.crosstab(train_df.Prod_Category_1,train_df.gender).plot(kind="bar",figsize=(20,6))
plt.title('Monthly freuency to see product by genders')
plt.xlabel('Month')
plt.legend(["Female", "Male"])
plt.ylabel('Frequency')
plt.show()


# In[18]:


pd.crosstab(train_df.month,train_df.gender).plot(kind="bar",figsize=(20,6))
plt.title('Monthly freuency to see product by genders')
plt.xlabel('Month')
plt.legend(["Female", "Male"])
plt.ylabel('Frequency')
plt.show()


# In[19]:


pd.crosstab(train_df.week,train_df.gender).plot(kind="bar",figsize=(20,6))
plt.title('Weekly freuency to see product by genders')
plt.xlabel('Week')
plt.legend(["Female", "Male"])
plt.ylabel('Frequency')
plt.show()


# In[20]:


plt.figure(figsize=(7, 3))
sns.boxplot(train_df["hour"])
plt.show()


# In[21]:


varlist = ['gender']
def num_map(x):
    return x.map({'female': 0, "male": 1})
train_df[varlist] = train_df[varlist].apply(num_map)


# In[22]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for x in train_df.columns:
    if train_df[x].dtype == type(object):
        train_df[x] = train_df[x].fillna('NaN')
        test_df[x] = test_df[x].fillna('NaN')
        encoder = LabelEncoder()
        encoder.fit(list(set(list(train_df[x]) + list(test_df[x]))))
        train_df[x] = encoder.transform(train_df[x])
        test_df[x] = encoder.transform(test_df[x])


# In[23]:


# Splitting training dataset into train and test
X = train_df.copy().drop('gender', axis=1).values
y = train_df['gender']


# ## Model Building

# In[24]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test_v = sc.transform(test_df.copy().values)


# ## XGBoost

# In[25]:


# XGB Classifier
from xgboost import XGBClassifier

xgb = XGBClassifier( learning_rate =0.1,
 n_estimators=112,
 max_depth=9,
 min_child_weight=5,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.6,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=13,
 reg_lambda=5,
# max_delta_step=1,
 alpha=0,
 base_score=0.5,
 seed=1029)

xgb.fit(X_train, y_train)

# Predicting the Test set results
Y_predicted = xgb.predict(X_test)  

# Accuracy of XGB model
accuracy_xgb = round(xgb.score(X_train, y_train) * 100, 2)
print("Accuracy score of XGB algorithm is:", accuracy_xgb)


# In[27]:


# Predicting the Test set results
test_pred = xgb.predict(test_v)


# In[28]:


# load session_id of test dataset
test_session_id = pd.read_csv('test_Yix80N0.csv')['session_id']
print(test_session_id.shape)


# In[29]:


# save results to csv
submission_file = pd.DataFrame({'session_id': test_session_id, 'gender': test_pred})
submission_file = submission_file[['session_id','gender']] 
varlist = ['gender']
def num_map(x):
    return x.map({0:'female', 1:'male'})
submission_file[varlist] = submission_file[varlist].apply(num_map)
submission_file.to_csv('Final_Solution.csv', index=False)

