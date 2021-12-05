#!/usr/bin/env python
# coding: utf-8

# # FINAL PROJECT for CS 634 
# ## Name: Veena Chaudhari
# ## Topic: Predicting whether an individual is obese or not  based on their eating habits and physical condition 

# Github link: https://github.com/vac38/Classification_of_obesity.git

# 
# 
# Link to dataset: https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+
# 

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# In[2]:


# Storing data in a pandas dataframe 
ObesityData = pd.read_csv("/Users/veena/Desktop/DM_final_project/ObesityDataSet.csv")


# ## Exploratory Data analysis

# In[3]:


# To display the data type for each feature/Atrribute
ObesityData.info()


# In[4]:


ObesityData


# ### 1) Renaming columns in data

# In[5]:


#Renaming columns in data
ObesityData.columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
       'high_caloric_food', 'vegetables_consumption', 'main_meals', 'food_between_meals', 'SMOKE', 'Daily_water', 'Calories_consumption', 'physical_activity', 'technology_devices',
       'Alcohol_consumption', 'Transportation_used', 'Obesity']


# ### 2) converting label values to binary
# 
# Since the task for this project is to perform binary classification , the labels were categorized in to Normal or Obese using the following distinction: 
# 
# Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II → Categorized as ‘NORMAL’
# 
# Obesity Type II and Obesity Type III →  categorized as ‘OBESE’
# 

# In[6]:


# Get all values present in the label column of dataset
ObesityData['Obesity'].unique()


# In[7]:


# convert to labels to Normal and Obese
ObesityData['Obesity'].replace({'Normal_Weight': 'Normal','Overweight_Level_I':'Normal' , 'Overweight_Level_II':'Normal', 'Insufficient_Weight':'Normal', 'Obesity_Type_I':'Obese','Obesity_Type_II':'Obese','Obesity_Type_III':'Obese'}, inplace= True)


# In[8]:


# Only two labels: Normal and Obese
ObesityData['Obesity'].unique()


# In[9]:


# Checking for imbalance in data 
ObesityData['Obesity'].value_counts()


# The distribution of each class with the labels shows that the data is not balanced since 1139 records belong to ‘Normal’ class and 972 to ‘Obese’ class and their ratio is ~1.17

# ### 3) Shape of Data

# In[10]:


ObesityData.shape


# ### 4) Check for null values

# In[11]:


#Check if there are any missing values
print("Column wise missing values in Data\n",ObesityData.isnull().sum())
sns.heatmap(ObesityData.isnull(), yticklabels=False)


# ### 5) Age group of people in Dataset

# In[12]:


sns.displot(ObesityData['Age'] , bins = 20, kde=True)
print('Average age: ',ObesityData['Age'].mean())


# The Age group of most of the participants in this study is 15 to 28 years with average age of 24 years

# ### 6) Average height and weight for the males and females

# In[13]:


sns.set()
fig = plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
sns.boxplot(x='Gender', y='Height', data=ObesityData)
plt.subplot(1,2, 2)
sns.boxplot(x='Gender', y='Weight', data=ObesityData)


# The above box plots show that average height for males is more than females.
# 
# Average weight of males is more than that of females

# ### 7) Relation plot for weight ,height , genders and obesity

# In[14]:


subdf1 = ObesityData.iloc[:,[0,2,3,16]]
sns.relplot(x="Height", y="Weight", hue="Obesity",style="Gender", data=subdf1)


# Th above plot shows how height and weight influence obesity. 
# 1) People with higher weights tend to be more obese 
# 
# 2) Obesity does determined by ratio of height and weight.

# ## Data Preprocessing

# ### 1) Label Encoding
# Since Classifiers cannot handle label data directly, label encoding is used.

# In[15]:


ObesityData.head(10)


# In[16]:


lenc = LabelEncoder()
ObesityData['food_between_meals'] = lenc.fit_transform(ObesityData['food_between_meals'])
ObesityData['SMOKE'] = lenc.fit_transform(ObesityData['SMOKE'])
ObesityData['Calories_consumption'] = lenc.fit_transform(ObesityData['Calories_consumption'])
ObesityData['Alcohol_consumption'] = lenc.fit_transform(ObesityData['Alcohol_consumption'])
ObesityData['Gender'] = lenc.fit_transform(ObesityData['Gender'])
ObesityData['family_history_with_overweight'] = lenc.fit_transform(ObesityData['family_history_with_overweight'])
ObesityData['high_caloric_food'] = lenc.fit_transform(ObesityData['high_caloric_food'])
ObesityData['Transportation_used'] = lenc.fit_transform(ObesityData['Transportation_used'])
ObesityData['Obesity'] = lenc.fit_transform(ObesityData['Obesity'])


# In[17]:


ObesityData.head(10)


# ### 2) Correlation between different features

# In[18]:


#Correlation matrix
ObesityData.corr()

#Correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(ObesityData.corr(), annot = True)

# No two features are highly correlated


# ### 3) Splitting the data in to features(X) and Labels(Y) 

# In[19]:


X_n = ObesityData[['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
       'high_caloric_food', 'vegetables_consumption', 'main_meals', 'food_between_meals', 'SMOKE', 'Daily_water', 'Calories_consumption', 'physical_activity', 'technology_devices',
       'Alcohol_consumption','Transportation_used']].values
Y = ObesityData['Obesity']


# ### 4) Normalization of Data

# The range of values for each feature are different. For example weight ranges from 39 kgs to 173 kgs and gender has only two values: 0 and 1. Therefore to convert all feature values between 0 and 1 , normalization is performed.

# In[22]:


#returns a numpy array with normalized values for X
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X_n)


# # Machine Learning models

# In[23]:


from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy import mean


# In[24]:


def calc_evaluation_metrics(TN,FP,FN,TP):

    # Sensitivity (recall o true positive rate)
    Sensitivity = TP/(TP+FN)
    # Specificity(true negative rate)
    Specificity = TN/(TN+FP) 
    # Precision(positive predictive value)
    Precision = TP/(TP+FP)
    # Error Rate
    Err = (FP + FN)/(TP + FP + FN + TN)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # False positive rate
    FPR = FP/(FP+TN)
    # False Discovery Rate
    FDR = FP / (FP + TP)
    # False negative rate
    FNR = FN/(TP+FN)
    # Overall accuracy
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    #F1_score 
    F1_score = (2 * TP)/(2 *( TP + FP + FN))
    #Balanced Acuuracy(BACC)
    BACC = (Sensitivity + Specificity)/2
    #True Skills Statistics(TSS)
    TSS = (𝑇𝑃/(𝑇𝑃+𝐹𝑁)) - (𝐹𝑃/(𝐹𝑃+𝑇𝑁))
    #Heidke Skill Score (HSS)
    num = 2 * ((TP*TN)-(FP*FN))
    denom = ((𝑇𝑃 + 𝐹𝑁) * ((𝐹𝑁+𝑇𝑁)+(TP+FP))* (𝐹𝑃+𝑇𝑁))
    HSS = num / denom
    return Accuracy,Sensitivity, Specificity, Precision,F1_score, Err, NPV, FPR,FDR,FNR,BACC,TSS,HSS


# In[25]:


def kfold_split(X,Y,train_index, test_index):
    X_train, X_test = X[train_index], X[test_index] 
    y_train, y_test = Y[train_index], Y[test_index]
    return X_train, X_test,y_train, y_test

def c_matrix (y_test, LR_pred, m, i):
    c_matrix=confusion_matrix(y_test, LR_pred).ravel()
    TN, FP, FN, TP = c_matrix[0],c_matrix[1], c_matrix[2],c_matrix[3]
    Accuracy,Sensitivity, Specificity, Precision,F1_score, Err, NPV, FPR,FDR,FNR,BACC,TSS,HSS = calc_evaluation_metrics(TN,FP,FN,TP)
    metrics = [m,i, Accuracy,Sensitivity, Specificity, Precision,F1_score, Err, NPV, FPR,FDR,FNR,BACC,TSS,HSS]
    return metrics

def logistic(X_train, X_test,y_train, y_test):
    model_LR = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
    model_LR.fit(X_train, y_train)
    LR_pred = model_LR.predict(X_test)
    return LR_pred


def decision_tree(X_train, X_test,y_train, y_test):
    decisiontree_model = DecisionTreeClassifier(random_state=0)
    decisiontree_model.fit(X_train,y_train)
    dt_pred = decisiontree_model.predict(X_test)
    return dt_pred

def random_forest(X_train, X_test,y_train, y_test):
    randomforest_model = RandomForestClassifier(max_depth = 100,  max_features= 3, min_samples_leaf= 3)
    randomforest_model.fit(X_train,y_train)
    rt_pred = randomforest_model.predict(X_test)
    return rt_pred

    


# ### Training and testing three diffrent machine learning models: Logistic Reression, Decision Tree and Random Forest

# In[26]:


kf = KFold(n_splits=10,random_state=None, shuffle = True) 
model_acc_LR = []
model_acc_DT = []
model_acc_RF = []

# LR = pd.DataFrame(columns =['model','fold','Accuracy','Sensitivity', 'Specificity', 'Precision', 'F1_score','Error rate', 'Negative predictive value', 'False positive rate', 'False Discovery Rate', 'False negative rate', 'Balanced Accuracy', 'True Skill Statistics','Heidke Skill Score'])
i = 1
for train_index, test_index in kf.split(X):
    # Sets of train and test
    X_train, X_test,y_train, y_test = kfold_split(X,Y, train_index, test_index)
    # models and prediction
    LR_pred = logistic(X_train, X_test,y_train, y_test)
    DT_pred = decision_tree(X_train, X_test,y_train, y_test)
    RF_pred = random_forest(X_train, X_test,y_train, y_test)

    #Evaluation : Logistic regression
    metric_LR = c_matrix(y_test, LR_pred, 'Logistic Regression', i)
    model_acc_LR.append(metric_LR)
    
    #Evaluation : Decision Tree
    metric_DT = c_matrix(y_test, DT_pred, 'Decision Tree', i)
    model_acc_DT.append(metric_DT)
    
    #Evaluation : Random Forest
    metric_RF = c_matrix(y_test, RF_pred, 'Random Forest', i)
    model_acc_RF.append(metric_RF)

    i += 1  
# Storing Data in Datframe
LR_metrics = pd.DataFrame(model_acc_LR, columns =['model','fold','Accuracy','Sensitivity', 'Specificity', 'Precision', 'F1_score','Error rate', 'Negative predictive value', 'False positive rate', 'False Discovery Rate', 'False negative rate', 'Balanced Accuracy', 'True Skill Statistics','Heidke Skill Score'])
LR_metrics.loc['Mean'] = LR_metrics.mean()

DT_metrics = pd.DataFrame(model_acc_DT, columns =['model','fold','Accuracy','Sensitivity', 'Specificity', 'Precision', 'F1_score','Error rate', 'Negative predictive value', 'False positive rate', 'False Discovery Rate', 'False negative rate', 'Balanced Accuracy', 'True Skill Statistics','Heidke Skill Score'])
DT_metrics.loc['Mean'] = DT_metrics.mean()

RF_metrics = pd.DataFrame(model_acc_RF, columns =['model','fold','Accuracy','Sensitivity', 'Specificity', 'Precision', 'F1_score','Error rate', 'Negative predictive value', 'False positive rate', 'False Discovery Rate', 'False negative rate', 'Balanced Accuracy', 'True Skill Statistics','Heidke Skill Score'])
RF_metrics.loc['Mean'] = RF_metrics.mean()


# In[27]:


# Results for logistic regression performed on Obesity data using 10-fold cross validation
LR_metrics


# In[28]:


# Results for Decision tree performed on Obesity data using 10-fold cross validation

DT_metrics


# In[35]:


# Results for Random forest performed on Obesity data using 10-fold cross validation

RF_metrics


# # Deep Learning

# ### LSTM

# In[30]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[ ]:


model_acc_lstm = []
i = 1
for train, test in kf.split(X):
    X_train, X_test = X[train], X[test] 
    y_train, y_test = Y[train], Y[test]
  # create model
    model = Sequential()
    model.add(LSTM(200, activation='relu',input_shape=(1,16)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='sigmoid'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    X_train_new = X_train.reshape((X_train.shape[0],1, X_train.shape[1]))
    X_test_new = X_test.reshape((X_test.shape[0],1, X_test.shape[1]))
    model.fit(X_train_new,y_train, epochs = 100, batch_size = 32, verbose=0)    
    # predict on the model
    predval = model.predict(X_test_new).flatten()
    predval_new = np.where(predval > 0.5, 1, 0)
    #Evalute the model
    metric_lstm = c_matrix(y_test, predval_new, 'LSTM', i)
    model_acc_lstm.append(metric_lstm)
    i += 1
    
    
LSTM_metrics = pd.DataFrame(model_acc_lstm, columns =['model','fold','Accuracy','Sensitivity', 'Specificity', 'Precision', 'F1_score','Error rate', 'Negative predictive value', 'False positive rate', 'False Discovery Rate', 'False negative rate', 'Balanced Accuracy', 'True Skill Statistics','Heidke Skill Score'])
LSTM_metrics.loc['Mean'] = LSTM_metrics.mean()


# In[32]:


# Results for LSTM performed on Obesity data using 10-fold cross validation

LSTM_metrics


# In[33]:


lr = pd.DataFrame(LR_metrics.iloc[10:,2:])
dt = pd.DataFrame(DT_metrics.iloc[10:,2:])
rf = pd.DataFrame(RF_metrics.iloc[10:,2:])
lstm = pd.DataFrame(LSTM_metrics.iloc[10:,2:])
k = [lr,dt,rf,lstm]
ALL_models = pd.concat(k)
obesity_predictions = ALL_models.set_axis(['Linear Regression', 'Decision Tree', 'Random Forest','LSTM'], axis=0)


# ## Conclusion 

# In[34]:


obesity_predictions


# ##  Which algorithm performs better ?

# 
# On comparing the accuracy, it is evident that Random forest outperforms all other models and therfore the best model for predicting obesity given the this dataset. 
# 
# The Random forest Algorithm performs better than all the other models, because random forest can handle classification tasks with all kinds of input features and also with minimal preprocessing.
# 
