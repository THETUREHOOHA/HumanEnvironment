# -*- coding: utf-8 -*-
"""Copy of beautiful/wealthier_score

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ywZ-KI9YXRo4yvpzeDBwJC4zAhXtokO3

# Load Score Model
## Load Training Data

Load the CSV with Scores (label/y/dependent variable) and PspNet Results (X feature space/ independent variables)
"""

### Read Training Dataset
import pandas as pd
df= pd.read_csv("/training_datasets_place_pulse2.0/beautiful_score.csv",encoding='utf-8') #, engine='python''gb2312','utf-8'
print(df.shape)
df.head(2)

df.columns

"""# Load the data we want to make predictions"""

import pandas as pd
df_MI= pd.read_csv("Milan_street_element_seg_result.csv",encoding='utf-8', engine='python') #'gb2312','utf-8'

columns_to_remove = df_MI.columns[33:40]
df_MI = df_MI.drop(columns_to_remove, axis=1)

print(df_MI.shape)
df_MI.head(3)

"""### Lookinto the data structure"""

### Lookinto the data structure

df_PP=df.copy()

df_PP_Q=df_PP.iloc[:, :3]

df_PP_PSP=df_PP.iloc[:,3:41]

for i in [df_PP_Q,df_PP_PSP]:
    print(i.shape,i.columns)

### Split PSPNET result for Prediction Overall Score as xPRE
df_MI_PSP=df_MI.iloc[:, 4:]
print('Will predict based on %s dimensions of features'%(df_MI_PSP.shape[1]))
print('Will predict based on %s dimensions of features'%(df_MI_PSP.shape[1]))

### 0.Get the columns for Analysis in consistent order for both Training data and Prediction data
Diff=[]
for col_name in df_MI.columns:
    if col_name not in df_PP_PSP.columns:
        Diff.append(col_name)

print('Different columns are %s'%(Diff),'\n')
reindex_columns=list(set(df_MI.columns)-set(Diff))
reindex_columns.sort()

print("Selected columns:",len(reindex_columns),reindex_columns,'\n')

### 1.Split PSPNET result from Prediction Data as df_MI_PSP
df_MI_PSP = df_MI[reindex_columns]

### 2.Split PSPNET result from Training Data as df_PP_PSP
df_PP_PSP = df_PP_PSP[reindex_columns]


print("Reshape xTr to %s, and xPre to %s dimensions"%(len(df_PP_PSP.columns),len(df_MI_PSP.columns)),'\n')

if df_PP_PSP.columns.all() == df_MI_PSP.columns.all():
    print("Dimensions for Traning and Prediction Dataset are Matched")

"""# 1. Train/Test Data Split"""

### Divide data for training & testing on SafetyScore
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

X = df_PP_PSP.values
Y = df_PP_Q.iloc[:, 2].values
xTr, xTe, yTr, yTe = train_test_split(X, Y, test_size=0.20)

# xTr-yTr, xTe-yTe, y=beta*x

print(xTr.shape,xTe.shape)

# Function Syntax
def multiplier (a,b):
    c=a*b
    return c

multiplier(3,4)

"""# Model 1. KNN
-------

#### 1.1 Define the complete KNN function from scratch with three sub functions l2distance, findknn, and knnclassifier.
These functions will do following tasks:

(a) calculate distance,
(b) find k nearest neighbors,
(c) get the label or average value
"""

# Commented out IPython magic to ensure Python compatibility.
### ***********************Preparation_K-NN Machine Learning Codes *************************
import numpy as np
from scipy.stats import mode
import sys
# %matplotlib notebook
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

print('You\'re running python %s' % sys.version.split(' ')[0])

# Computes the Euclidean distance matrix.
def l2distance(X,Z=None):
    if Z is None:
        Z=X;
    n,d1=X.shape
    m,d2=Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"
    Xsqsum = np.tile(np.sum(np.square(X),axis=1),(m,1)).T
    Zsqsum = np.tile(np.sum(np.square(Z), axis=1),(n,1))
    innprod = 2*np.dot(X, Z.T)
    D = np.add(Xsqsum,Zsqsum)
    D = np.subtract(D,innprod)
    D = np.sqrt(np.abs(D))
    D[D<0]=0
    return D

# Finds the k nearest neighbors of xTe in xTr
def findknn(xTr,xTe,k):
    from_test_train = l2distance(xTe,xTr)
    matrix_sortby_distance = from_test_train.argsort()
    indices= (matrix_sortby_distance[:,0:k]).T
    dist_matrix = np.sort(from_test_train,axis=1)
    dists=(dist_matrix[:,0:k]).T
    return indices, dists

def analyze(kind,truth,preds):
    truth = truth.flatten()
    preds = preds.flatten()
    if kind == 'abs':
        print("abs analysis")
        abs_loss=0
        abs_loss=(np.sum(np.abs(truth-preds)))/len(preds)
        output=abs_loss
    elif kind == 'acc':
        print("acc analysis")
        count=0
        for i in range(len(preds)):
            if preds[i]==truth[i]:
                count+=1
            output= (float(count)/float(len(preds)))
    return output

import collections
from scipy import stats
#***************************************

def knnclassifier_fre(xTr,yTr,xTe,k):
    yTr = yTr.flatten()
    [indices,dists]=findknn(xTr,xTe,k)
    preds=[]
    for i in range(indices.shape[1]):
        temp=[]
        for j in indices[:,i]:
            temp.append(yTr[j])
        # 1.get average value
        #d=np.mean(temp)
        # 2.look for most frequent values
        d=collections.Counter(temp).most_common(1)[0][0]
        preds.append(d)
    return np.array(preds)

def knnclassifier_ave(xTr,yTr,xTe,k):
    yTr = yTr.flatten()
    [indices,dists]=findknn(xTr,xTe,k)
    preds=[]
    for i in range(indices.shape[1]):
        temp=[]
        for j in indices[:,i]:
            temp.append(yTr[j])
        # 1.get average value
        d=np.mean(temp)
        # 2.look for most frequent values
        #d=collections.Counter(temp).most_common(1)[0][0]
        preds.append(d)
    return np.array(preds)

"""#### Print matplotlib graph in line"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

"""## 1.2 Run the KNN function and compare loss"""

### Compute RMSE & R2 for KNN K in [1,100]
pDist={'Dis':{},'Uni':{}}
wghts={'Dis':'distance','Uni':'uniform'}

# **************************Start Timer*****************************************
from datetime import datetime
startTime = datetime.now() # initiate timer

for key in wghts.keys():
    w=wghts[key]
    for i in range(1,3):
        r2List,rmseList=([],[])
        for k in range(1,50):
            knnRegressor = KNeighborsRegressor(n_neighbors=k,p=i,weights=w) # weights=distance/uniform
            knnRegressor.fit(xTr, yTr)
            yPr=knnRegressor.predict(xTe)
            rmse=np.sqrt(np.sum((yPr-yTe)**2)/len(yPr))
            rmseList.append(rmse)
            r2=r2_score(yTe,yPr)
            r2List.append(r2)
        pDist[key][i]=(rmseList,r2List)
    #print(k,round(max(yPr-yTe),2),round(min(yPr-yTe),2),round(np.mean(abs(yPr-yTe)),2),ls)
# *************Stop Timer***
print("runtime:",datetime.now()-startTime) # stop timer

"""## 1.3 Visualize the modeling process using R2, MSRE"""

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(9, 4))
sub1 = fig.add_subplot(121)
alp=0.9
lgth=len(pDist['Dis'][1][0])
sub1.plot(range(lgth),pDist['Dis'][1][0],label="Weighted_L1",linewidth=3,alpha=alp)
sub1.plot(range(lgth),pDist['Dis'][2][0],label="Weighted_L2",alpha=alp)
# sub1.plot(range(lgth),pDist['Dis'][3][0],label="Weighted_L3",alpha=alp)
# sub1.plot(range(lgth),pDist['Dis'][4][0],label="Weighted_L4",alpha=alp)
sub1.plot(range(lgth),pDist['Uni'][1][0],label="Uniform_L1",alpha=alp)
sub1.plot(range(lgth),pDist['Uni'][2][0],label="Uniform_L2",alpha=alp)
# sub1.plot(range(lgth),pDist['Uni'][3][0],label="Uniform_L3",alpha=alp)
# sub1.plot(range(lgth),pDist['Uni'][4][0],label="Uniform_L4",alpha=alp)
sub1.set_xlabel('K Nearest Neighbor',weight='bold')
sub1.set_ylabel('Root-mean-square-error (RMSE)',weight='bold')
sub1.set_title('KNN Loss Function')
sub1.grid('True')
# sub1.set_yticks(np.arange(1.4, 2, 0.05))
sub1.set_xticks(np.arange(0, 50, 3))
plt.gca().legend()
fig.tight_layout()

sub2 = fig.add_subplot(122)
sub2.plot(range(lgth),pDist['Dis'][1][1],label="Weight_L1",linewidth=3)
sub2.plot(range(lgth),pDist['Dis'][2][1],label="Weight_L2")
# sub2.plot(range(lgth),pDist['Dis'][3][1],label="Weight_L3")
# sub2.plot(range(lgth),pDist['Dis'][4][1],label="Weight_L4")
sub2.plot(range(lgth),pDist['Uni'][1][1],label="Unifirom_L1")
sub2.plot(range(lgth),pDist['Uni'][2][1],label="Unifirom_L2")
# sub2.plot(range(lgth),pDist['Uni'][3][1],label="Unifirom_L3")
# sub2.plot(range(lgth),pDist['Uni'][4][1],label="Unifirom_L4")
sub2.set_xlabel('K Nearest Neighbor',weight='bold')
sub2.set_ylabel('R Square',weight='bold')
sub2.set_title('KNN Loss Function')
sub2.grid('True')
# plt.yticks(np.arange(-0.1, 0.4, 0.05))
plt.xticks(np.arange(0, 50, 3))
plt.gca().legend()
fig.tight_layout()
plt.show()

"""## 1.4 Analyze the best solution configuration"""

### Analyze the best solution
error=np.inf
for weight in pDist.keys():
    row = []
    print('Weight Type',weight)
    for key in pDist[weight].keys():
        row.append(pDist[weight][key][0])
        #print(weight,key)
    df = pd.DataFrame()
    df = pd.DataFrame(row)
    df2array = df.values
    l = np.min(df2array)
    w=weight
    #print(minloss)
    ind = np.unravel_index(np.argmin(df2array, axis=None), df2array.shape)
    d,k = ind[0]+1,ind[1]+1
    print('Loss',l,'K-NN',k,'Distance',d,'Weight',w)
    if l<error:
        error=l
        best_w,best_d,best_k,best_l=w,d,k,l
print("*** Best Model Setting*** min RMSE",best_l,", @K=",best_k,", l_dist=",best_d,", weight type:",wghts[best_w])

"""## 1.5 Apply Model to Make Prediction


"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor

# Set up Prediction data xPre
xPre = df_MI_PSP.values
print(xPre.shape)

# Create an imputer to fill missing values with mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to your training data and transform the prediction data
xTr_imputed = imputer.fit_transform(xTr)
xPre_imputed = imputer.transform(xPre)

# Make Prediction Based on Best Fit Model
knnRegressor = KNeighborsRegressor(n_neighbors=best_k, p=best_d, weights=wghts[best_w])  # weights=distance/uniform
knnRegressor.fit(xTr_imputed, yTr)
yPre = knnRegressor.predict(xPre_imputed)

# Add the prediction to the DataFrame
yPre_List = list(yPre)
column_values = pd.Series(yPre_List)
df_MI.insert(loc=3, column='depressing_score', value=column_values)
print(df_MI.columns, df_MI.shape)
df_MI.head(1)

"""# 1.6 Save prediction Results"""

# Save Results
df_MI.to_csv("prediction_beautiful_score.csv", index=False, encoding='utf-8')

"""# Model 2. SVM"""

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

### Support Vector Machine for Regression

from sklearn.svm import SVR
import numpy as np

from datetime import datetime
startTime = datetime.now() # initiate timer

rngC=range(16,20) #Penalty parameter C of the error term.
pDist={}
for j in [0.1,0.01,0.001,0.0001]:
    r2List,rmseList=([],[])
    for i in rngC:
        clf = SVR(gamma='scale', C=i, epsilon=j)
        clf.fit(xTr,yTr)
        yPr=(clf.predict(xTe))
        rmse=np.sqrt(np.sum((yPr-yTe)**2)/len(yPr))
        rmseList.append(rmse)
        r2=r2_score(yTe,yPr)
        r2List.append(r2)
    pDist[j]=(r2List,rmseList)
#print(pDist)

### Plot the result
fig = plt.figure(figsize=(9, 4))
sub1 = fig.add_subplot(121)
alp=0.9
sub1.plot(rngC,pDist[0.1][0],label="eps=.1",linewidth=3,alpha=alp)
sub1.plot(rngC,pDist[0.01][0],label="eps=.01",alpha=alp)
sub1.plot(rngC,pDist[0.001][0],label="eps=.001",alpha=alp)
sub1.plot(rngC,pDist[0.0001][0],label="eps=.0001",alpha=alp)
sub1.set_xlabel('Penalty parameter C of the error term.',weight='bold')
sub1.set_ylabel('R Square',weight='bold')
sub1.set_title('SVM Regression')
sub1.grid('True')
# sub1.set_yticks(np.arange(1.4, 2, 0.05))
#sub1.set_xticks(np.arange(0, 0.105, 0.01))
plt.gca().legend()
fig.tight_layout()

sub2 = fig.add_subplot(122)
sub2.plot(rngC,pDist[0.1][1],label="eps=.1",linewidth=3,alpha=alp)
sub2.plot(rngC,pDist[0.01][1],label="eps=.01",alpha=alp)
sub2.plot(rngC,pDist[0.001][1],label="eps=.001",alpha=alp)
sub2.plot(rngC,pDist[0.0001][1],label="eps=.0001",alpha=alp)
sub2.set_xlabel('Penalty parameter C of the error term.',weight='bold')
sub2.set_ylabel('Root-mean-square-error (RMSE)',weight='bold')
sub2.set_title('SVM Regression')
sub2.grid('True')
# sub1.set_yticks(np.arange(1.4, 2, 0.05))
#sub1.set_xticks(np.arange(0, 0.105, 0.01))
plt.gca().legend()
fig.tight_layout()

endTime = datetime.now()
print('Duration: {}'.format(endTime - startTime))

print(pDist)

"""#1.2 Apply model to prediction"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

from sklearn.svm import SVR
import numpy as np

from datetime import datetime
startTime = datetime.now() # initiate timer
xPre = df_MI_PSP.values
yPre = clf.predict(xPre)


# Add the prediction to  DataFrame

yPre_List = list(yPre)
column_values = pd.Series(yPre_List)
df_MI.insert(loc=3, column='beautiful_score', value=column_values)
print(df_MI.columns, df_MI.shape)
df_MI.head(1)

# Save Results
df_MI.to_csv("prediction_beautiful_score.csv", index=False, encoding='utf-8')

"""## Model 3.  Random Forest"""

### Random Forest
#model= RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

from datetime import datetime
startTime = datetime.now() # initiate timer

RandomForest = RandomForestRegressor()
RandomForest.fit(xTr,yTr)
yPr=RandomForest.predict(xTe)

rmse=np.sqrt(np.sum((yPr-yTe)**2)/len(yPr))
r2=r2_score(yTe,yPr)
mae=np.sum(np.sqrt((yPr-yTe)**2))/len(yPr)
print(rmse,r2,mae,RandomForest.get_params)

maxDepthList=range(5,20) #Penalty parameter C of the error term.
Results={}
for est in range(10,20):
    r2List,rmseList,maeList=([],[],[])
    for dep in maxDepthList:
        clf = RandomForestRegressor(max_depth=dep, random_state=0, n_estimators=est)
        clf.fit(xTr,yTr)
        yPr=clf.predict(xTe)
        rmse=np.sqrt(np.sum((yPr-yTe)**2)/len(yPr))
        rmseList.append(rmse)

        r2=r2_score(yTe,yPr)
        r2List.append(r2)

        mae=np.sum(np.sqrt((yPr-yTe)**2))/len(yPr)
        maeList.append(mae)

    Results[est]=(r2List,rmseList,maeList)
#print(pDist)

### Plot the result
fig = plt.figure(figsize=(15, 5))
alp=0.9

for i in Results.keys():
    sub1 = fig.add_subplot(131)
    sub1.plot(maxDepthList,Results[i][0],label="n_estimators="+str(i),linewidth=1,alpha=alp)
    sub1.set_xlabel('Max Depth',weight='bold')
    sub1.set_ylabel('R Square',weight='bold')
    sub1.set_title('Random Forest')
    sub1.grid('True')
    #sub1.set_yticks(np.arange(1.4, 2, 0.05))
    #sub1.set_xticks(np.arange(0, 0.105, 0.01))
    plt.gca().legend()
    fig.tight_layout()

    sub2 = fig.add_subplot(132)
    sub2.plot(maxDepthList,Results[i][1],label="n_estimators="+str(i),linewidth=1,alpha=alp)
    sub2.set_xlabel('Max Depth',weight='bold')
    sub2.set_ylabel('Root-mean-square-error (RMSE)',weight='bold')
    sub2.set_title('Random Forest')
    sub2.grid('True')
    #sub1.set_yticks(np.arange(1.4, 2, 0.05))
    #sub1.set_xticks(np.arange(0, 0.105, 0.01))
    plt.gca().legend()
    fig.tight_layout()

    sub3 = fig.add_subplot(133)
    sub3.plot(maxDepthList,Results[i][2],label="n_estimators="+str(i),linewidth=1,alpha=alp)
    sub3.set_xlabel('Penalty parameter C of the error term.',weight='bold')
    sub3.set_ylabel('Mean Absolute Error (MAE)',weight='bold')
    sub3.set_title('Random Forest')
    sub3.grid('True')
    #sub1.set_yticks(np.arange(1.4, 2, 0.05))
    #sub1.set_xticks(np.arange(0, 0.105, 0.01))
    plt.gca().legend()
    fig.tight_layout()

"""# 4. Compare Most Common ML Models"""

### Divide data for training & testing on Question 2-5
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

X = df_PP_PSP.values
Y = df_PP_Q.iloc[:, 2].values #Safety Score
xTr, xTe, yTr, yTe = train_test_split(X, Y, test_size=0.20)
yTr,yTe=(yTr.ravel(),yTe.ravel())

print(xTr.shape,xTe.shape)
print(type(yTr))

#KNN
from sklearn.neighbors import KNeighborsRegressor
KNN=KNeighborsRegressor().fit(xTr, yTr) #n_neighbors=2
yPr=KNN.predict(xTe)

r2=round(r2_score(yTe,yPr),2)
rmse=round(np.sqrt(np.sum((yPr-yTe)**2)/len(yPr)),2)
mae=round(np.sum(np.sqrt((yPr-yTe)**2))/len(yPr),2)
print("R2:%s, RMSE:%s, MAE:%s | KNN"%(r2,rmse,mae),'\n') # SVM_Reg.get_params

#SVM
SVM_Reg = SVR().fit(xTr,yTr)
yPr=SVM_Reg.predict(xTe)

r2=round(r2_score(yTe,yPr),2)
rmse=round(np.sqrt(np.sum((yPr-yTe)**2)/len(yPr)),2)
mae=round(np.sum(np.sqrt((yPr-yTe)**2))/len(yPr),2)
print("R2:%s, RMSE:%s, MAE:%s | SVM"%(r2,rmse,mae),'\n') # SVM_Reg.get_params

#Random Forest
from sklearn.ensemble import RandomForestRegressor
RandomForest = RandomForestRegressor().fit(xTr,yTr)
yPr=RandomForest.predict(xTe)

r2=round(r2_score(yTe,yPr),2)
rmse=round(np.sqrt(np.sum((yPr-yTe)**2)/len(yPr)),2)
mae=round(np.sum(np.sqrt((yPr-yTe)**2))/len(yPr),2)
print("R2:%s, RMSE:%s, MAE:%s | Random Forest"%(r2,rmse,mae),'\n')