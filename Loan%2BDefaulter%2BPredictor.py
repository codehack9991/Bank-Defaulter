
# coding: utf-8

# In[5]:


print('Importing packages...')
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# In[6]:


print('Importing models from scikit learn module..')
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.cross_validation import KFold   
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn import svm


# In[7]:


print('Reading data...')
dfTrain = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')


# In[8]:


dfTrain.apply(lambda x: sum(x.isnull()),axis=0)  # checking number of null values in each column


# In[9]:


null_columns=dfTrain.columns[dfTrain.isnull().any()]   # finding all rows which do not have null value in ratio
df_ratio_notnull=dfTrain[dfTrain["RATIO"].notnull()][null_columns]


# In[10]:


df_ratio_notnull.head()


# In[109]:


cols = ['DUE_MORTGAGE','VALUE','TJOB','DCL','CLT','CL_COUNT']   # imputing the data to make up for missing data
for col in cols: 
    print('Imputation with Median: %s' % (col))
    df_ratio_notnull[col].fillna(df_ratio_notnull[col].median(), inplace=True)
    dfTest[col].fillna(dfTest[col].median(), inplace=True)

    cols=['OCC']
for col in cols:
    print('Imputation with Zero: %s' % (col))
    df_ratio_notnull[col].fillna(0, inplace=True)
    dfTest[col].fillna(0, inplace=True)
    
    
    cols=['REASON']
for col in cols:
    print('Imputation with One: %s' % (col))
    df_ratio_notnull[col].fillna(1, inplace=True)
    dfTest[col].fillna(1, inplace=True)


# In[110]:


train_target = pd.DataFrame(df_ratio_notnull['RATIO'])   #training a model to learn the values of RATIO
train_cols=['DUE_MORTGAGE','VALUE','REASON','OCC','TJOB','DCL','CLT','CL_COUNT']
finalTrain=df_ratio_notnull[train_cols];
reg=LinearRegression()
reg.fit(finalTrain,train_target)


# In[111]:


dfTrain.head()


# In[112]:


dfTrain.describe()


# In[113]:


dfTrain['REASON'].value_counts()


# In[170]:


dfTrain['DEFAULTER'].value_counts()


# In[114]:


dfTrain['OCC'].value_counts()


# In[13]:


cols = ['AMOUNT','DUE_MORTGAGE','VALUE','TJOB','DCL','CLT','CL_COUNT']
for col in cols:
    print('Imputation with Median: %s' % (col))
    dfTrain[col].fillna(dfTrain[col].median(), inplace=True)
    dfTest[col].fillna(dfTest[col].median(), inplace=True)

    cols=['OCC']
for col in cols:
    print('Imputation with Zero: %s' % (col))
    dfTrain[col].fillna(0, inplace=True)
    dfTest[col].fillna(0, inplace=True)
    
    
    cols=['REASON']
for col in cols:
    print('Imputation with One: %s' % (col))
    dfTrain[col].fillna(1, inplace=True)
    dfTest[col].fillna(1, inplace=True)


# In[117]:


pred=reg.predict(dfTrain[train_cols])


# In[118]:


dfTrain['RATIO']=pred # putting the predicted valus of ratio in the dataframe


# In[119]:


dfTrain['log_amount'] = np.log(dfTrain['AMOUNT'])  # adding extra columns by performing log transformation
dfTrain['log_Due_Mortgage'] = np.log(dfTrain['DUE_MORTGAGE'])
dfTrain['log_value'] = np.log(dfTrain['VALUE'])


# In[120]:


train_target = pd.DataFrame(dfTrain['DEFAULTER'])


# In[121]:


train_cols=['log_amount','log_Due_Mortgage','log_value','REASON','OCC','TJOB','DCL','CLT','CL_COUNT','RATIO','CONVICTED','VAR_1','VAR_2','VAR_3']


# In[122]:


finalTrain=dfTrain[train_cols];


# In[123]:


X_train, X_test, y_train, y_test = train_test_split(np.array(finalTrain), np.array(train_target), test_size=0.30,random_state=59)


# In[124]:


clf = LogisticRegression()  # logistic regression


# In[125]:


clf.fit(X_train, y_train)


# In[126]:


y_pred = clf.predict(X_test)
print('Accuracy of logistic regression classifier on test set:')
print(clf.score(X_test, y_test))


# In[127]:


from sklearn.preprocessing import StandardScaler  # scaling the data so that neural network works better
scaler = StandardScaler()
scaler.fit(X_train)


# In[128]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[129]:


from sklearn.neural_network import MLPClassifier


# In[130]:


mlp = MLPClassifier(hidden_layer_sizes=(100,100,50,30,10),solver='adam',max_iter=1000,verbose=True)  # neural network


# In[131]:


mlp.fit(X_train,y_train)


# In[132]:


predictions=mlp.predict(X_test)


# In[133]:


print('Accuracy of neural network classifier on test set:')
print(mlp.score(X_test, y_test))


# In[134]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,predictions)


# In[135]:


dfTrain['annual_income']=dfTrain['AMOUNT']/dfTrain['RATIO'] # adding an extra column of annual_income


# In[136]:


dfTrain.describe()


# In[137]:


train_cols=['log_amount','log_Due_Mortgage','log_value','REASON','OCC','TJOB','DCL','CLT','CL_COUNT','RATIO','CONVICTED','annual_income','VAR_1','VAR_2','VAR_3']


# In[138]:


finalTrain=dfTrain[train_cols];


# In[139]:


X_train, X_test, y_train, y_test = train_test_split(np.array(finalTrain), np.array(train_target), test_size=0.30,random_state=59)


# In[140]:


clf=LogisticRegression()  #logistic Rgression __ again


# In[141]:


clf.fit(X_train,y_train)


# In[142]:


y_pred = clf.predict(X_test)
print('Accuracy of logistic regression classifier on test set:')
print(clf.score(X_test, y_test))


# In[143]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred)


# In[144]:


mlp_1= MLPClassifier(hidden_layer_sizes=(30,30,30),solver='adam')  # another neural network model with different number of hidden layers 


# In[145]:


mlp_1.fit(X_train,y_train)


# In[146]:


y_pred_1 = mlp_1.predict(X_test)
print('Accuracy of neural network classifier on test set:')
print(mlp_1.score(X_test, y_test))


# In[147]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_1)


# In[148]:


from sklearn import svm  # svm classifier


# In[149]:


svm_clf = svm.SVC(kernel='linear')


# In[150]:


svm_clf.fit(X_train,y_train)


# In[151]:


pred=svm_clf.predict(X_test)


# In[152]:


print('Accuracy of svm classifier on test set:')
print(svm_clf.score(X_test, y_test))


# In[153]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,pred)


# In[154]:


dfTest.head()


# In[160]:


dfTest['log_amount'] = np.log(dfTest['AMOUNT'])  # preparing the test data
dfTest['log_Due_Mortgage'] = np.log(dfTest['DUE_MORTGAGE'])
dfTest['log_value'] = np.log(dfTest['VALUE'])


# In[161]:


test_cols=['log_amount','log_Due_Mortgage','log_value','REASON','OCC','TJOB','DCL','CLT','CL_COUNT','RATIO','CONVICTED','VAR_1','VAR_2','VAR_3']


# In[162]:


final_test=dfTest[test_cols]


# In[164]:


predictions=mlp.predict(final_test) # using the neural network classifier for predictions


# In[168]:


import numpy as np
a = predictions
print(a)
unique_elements, counts_elements = np.unique(a, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))


# In[174]:


dfTest_2=dfTest.drop(['TEST_ID','VALUE','DUE_MORTGAGE','AMOUNT','log_amount','log_Due_Mortgage','log_value','REASON','OCC','TJOB','DCL','CLT','CL_COUNT','RATIO','CONVICTED','VAR_1','VAR_2','VAR_3'],axis=1)


# In[175]:


dfTest_2.head()


# In[176]:


dfTest_2['DEFAULTER']=predictions


# In[178]:


dfTest_2.describe()


# In[179]:


writer=pd.ExcelWriter('predictions.xlsx',engine='xlsxwriter')


# In[180]:


dfTest_2.to_excel(writer,'Sheet1')


# In[181]:


writer.save()

