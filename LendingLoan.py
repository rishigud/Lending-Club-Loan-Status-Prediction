# -*- coding: utf-8 -*-
#Natarajan

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dateutil.parser import parse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import seaborn as sns
#read the dataset using pandas read_csv method

data1 = pd.read_csv('LendingClub2012to2013.csv',header=1)

#see few rows and describe the dataset
data1.head()
data1.describe()

#check the data types of each column
data1.dtypes

#see how many numeric features are numeric
numeric_features = data1.select_dtypes(include=[np.number])
numeric_features.shape # 90 features are numeric

#see how many features are non-numeric
non_numeric_features = data1.select_dtypes(exclude=[np.number])
non_numeric_features.shape # 25 features are non numeric

non_numeric_features.dtypes #all the dtypes are in object form, look at the columns and convert them into
# string if it is a categorical variable and int,float if it is numeric

non_numeric_list = non_numeric_features.columns
non_numeric_list

#see the unique values
data1.id.unique()

#clean the data
data1.id[188181] = data1.id[188181].replace('Total amount funded in policy code 1: 2700702175','2700702175')
data1.id[188182] = data1.id[188182].replace('Total amount funded in policy code 2: 81866225','81866225')

#convert id column into float64
data1.id = data1.id.astype(np.float64)
data1.id.dtypes

data1 = data1[data1.grade.notnull()]

#con
data1.term.value_counts() #2 unique values
data1.term.isnull().sum()
data1.term = data1.term.fillna('0')
term= pd.get_dummies(data1.term,prefix='term')
data1 = pd.concat([data1,term],axis=1)
data1 = data1.drop('term',axis=1)
data1.int_rate.value_counts()

data1.grade.value_counts() # 7 unique values
data1.grade.isnull().sum()

data1.sub_grade.nunique() # 35
data1.emp_title.isnull().sum()
data1.emp_title = data1.emp_title.fillna('other')


#data cleaning
dic = {'years':'','year':'','10+ years':'10','< 1 year':'1','n/a':'0','10+':'10','< 1':'1'}

def empLength(x):
    for k,v in dic.items():
        x = x.replace(k,v)
    return x
   
data1.emp_length = data1.emp_length.astype(str)
data1.emp_length = data1.emp_length.apply(empLength)
data1.emp_length = data1.emp_length.apply(lambda x:  x.replace('s','') if 's' in x else x)
data1.emp_length.unique()
data1.emp_length.isnull().sum()
data1.emp_length = data1.emp_length.astype(np.float32)

data1.home_ownership.unique()
home_ownership= pd.get_dummies(data1.home_ownership,prefix='home_ownership')
data1 = pd.concat([data1,home_ownership],axis=1)
data1 = data1.drop('home_ownership',axis=1)

verification_status= pd.get_dummies(data1.verification_status,prefix='verification_status')
data1 = pd.concat([data1,verification_status],axis=1)
data1 = data1.drop('verification_status',axis=1)

data1.issue_d.isnull().sum()
data1.issue_d = data1.issue_d.apply(lambda x: parse(x))

data1.loan_status.unique()

data1.pymnt_plan.unique()

data1.title.nunique()

data1.revol_util.unique()

data1.initial_list_status.unique() 

data1.last_pymnt_d.isnull().sum()
data1.last_pymnt_d.nunique()
data1.last_pymnt_d = data1.last_pymnt_d.fillna(method='ffill')
data1.last_pymnt_d = data1.last_pymnt_d.apply(lambda x: parse(x))

data1.last_credit_pull_d.isnull().sum()
data1.last_credit_pull_d.nunique()
data1.last_credit_pull_d = data1.last_credit_pull_d.fillna(method='ffill')
data1.last_credit_pull_d = data1.last_credit_pull_d.apply(lambda x: parse(x))


y = data1.loc[:,['grade']].values

data1.int_rate = data1.int_rate.apply(lambda x: x.replace('%','') if '%' in x else x)
data1.int_rate = data1.int_rate.astype(np.float32)
data1.int_rate = data1.int_rate.apply(lambda x: x/100)

data1.loan_status.unique()
#****************************************************************************************#
# predicting grade column
X = data1.loc[:,['loan_amnt','funded_amnt','funded_amnt_inv','int_rate']]



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Random Forest


model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
model.fit(X_train,y_train)

y_pred_grade = model.predict(X_test)
y_pred_grade

cm_random_grade = confusion_matrix(y_test,y_pred_grade)
cm_random_grade
#Naive Bayes

model_naive = GaussianNB()
model_naive.fit(X_train,y_train)
y_pred_naive_grade = model_naive.predict(X_test)
y_pred_naive_grade
cm_naive_grade= confusion_matrix(y_test,y_pred_naive_grade)
cm_naive_grade
#Decision Tree

model_DT = DecisionTreeClassifier()
model_DT.fit(X_train,y_train)

y_pred_DT_grade = model_DT.predict(X_test)
y_pred_DT_grade

cm_DT_grade = confusion_matrix(y_test,y_pred_DT_grade)
cm_DT_grade

#****************************************************************************************#
##############################    Predicting Loan status      ############################
#****************************************************************************************#
data1.isnull().sum()

cols = data1.select_dtypes(include=[np.number])

cols_drop_nan = cols.dropna(axis=1)

cols_filled_mean = cols.fillna(cols.mean())

#######################  visualization ################################
cols_drop_nan.columns

######plotting box plot#######
plt.figure()
plt.boxplot(cols_drop_nan.loan_amnt)
plt.ylabel('Loan Amount')
plt.title('Box Plot for Loan Amount')

a = cols[['total_pymnt','total_rec_prncp','total_rec_int','total_rec_late_fee']]
a.index = data1['last_pymnt_d']

a = a.cumsum()
a.plot.hist(stacked=True, bins=15,title='Histogram plot')
plt.figure()
a.iloc[5].plot()


#### plotting top five received interested rate#####
def top_five(df,n=5,column='total_rec_int'):
    return df.sort_index(by=column)[-n:]


b = top_five(data1)
b.index=b['last_pymnt_d']
b = b[['total_pymnt','total_rec_prncp','total_rec_int','total_rec_late_fee']]
b = b.cumsum()
b.plot(kind='barh',title='Top Five Received Interest Rate')



####Distribution###

cols_box1=cols_drop_nan[['loan_amnt','total_pymnt','total_rec_prncp','total_rec_int']]
cols_box2=cols_drop_nan[['int_rate','emp_length','dti']]
color = dict(boxes='DarkGreen', whiskers='DarkOrange',medians='DarkBlue', caps='Gray')

cols_box1.plot.box(color=color, sym='r+',title = 'Distribution of features_1')
cols_box2.plot.box(color=color, sym='r+',title = 'Distribution of features_2')

###### area plot ####
df_fico = cols_drop_nan[['fico_range_low','fico_range_high','last_fico_range_high','last_fico_range_low']]
df_fico.plot.area()


sns.set(style="whitegrid", color_codes=True)
sns.stripplot(x="loan_amnt", y="loan_status", data=data1)
sns.boxplot(x="loan_amnt", y="loan_status", hue='initial_list_status',data=data1)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Logistic Regression @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
X_1 = data1.loc[:,cols_drop_nan.columns].values
y_lS = data1.loc[:,['loan_status']].values

X_train_2,X_test_2,y_train_2,y_test_2 = train_test_split(X_1,y_lS,test_size=0.25,random_state=42)

sc2 = StandardScaler()
X_train_2 = sc2.fit_transform(X_train_2)
X_test_2 = sc2.transform(X_test_2)

model_logistic = LogisticRegression()
model_logistic.fit(X_train_2,y_train_2)

y_pred_logistic = model_logistic.predict(X_test_2)
y_pred_logistic

cm_logistic = confusion_matrix(y_test_2,y_pred_logistic)
cm_logistic
cm_logistic.shape


report_logistic = metrics.classification_report(y_test_2,y_pred_logistic)
report_logistic

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Decision Tree @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

model_Decision = DecisionTreeClassifier()
model_Decision.fit(X_train_2,y_train_2)

y_pred_Decision = model_Decision.predict(X_test_2)
y_pred_Decision

cm_Decision = confusion_matrix(y_test_2,y_pred_Decision)
cm_Decision



report_Decision = metrics.classification_report(y_test_2,y_pred_Decision)
report_Decision

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Naive Bayes @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

model_naive = GaussianNB()
model_naive.fit(X_train_2,y_train_2)

y_pred_naive = model_naive.predict(X_test_2)
y_pred_naive

cm_naive = confusion_matrix(y_test_2,y_pred_naive)
cm_naive



report_naive = metrics.classification_report(y_test_2,y_pred_naive)
report_naive
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  KNN  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#



model_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model_KNN.fit(X_train_2,y_train_2)

y_pred_KNN = model_KNN.predict(X_test_2)
y_pred_KNN

cm_KNN = confusion_matrix(y_test_2,y_pred_KNN)
cm_KNN


report_KNN = metrics.classification_report(y_test_2,y_pred_KNN)
report_KNN

#@@@@@@@@@@@@@@@@@@@@@@@@@@@ Apllying k-fold Cross Validation @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
y_train_2 = y_train_2.reshape(141135,)
accuracies_Decision = cross_val_score(estimator = model_Decision, X = X_train_2, y = y_train_2, cv = 3,n_jobs=-1)
accuracies_Decision.mean()
accuracies_Decision.std()

accuracies_logistic = cross_val_score(estimator = model_logistic, X = X_train_2, y = y_train_2, cv = 3,n_jobs=-1)
accuracies_logistic.mean()
accuracies_logistic.std()

accuracies_Naive = cross_val_score(estimator = model_naive, X = X_train_2, y = y_train_2, cv = 3,n_jobs=-1)
accuracies_Naive.mean()
accuracies_Naive.std()

accuracies_KNN= cross_val_score(estimator = model_KNN, X = X_train_2, y = y_train_2, cv = 3,n_jobs=-1)
accuracies_KNN.mean()
accuracies_KNN.std()
