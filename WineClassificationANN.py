

 ''' CLASSIFICATION OF WINE QUALITY - BY AISWARYA SUNILKUMAR AND SREYAS K SREEKUMAR'''

 
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns


color_pal = sns.color_palette()
from warnings import filterwarnings
filterwarnings(action='ignore')

#loading dataset

path= 'winequality-red.csv'
wine = pd.read_csv(path)
print("Successfully Imported Data!")
wine.head()

print(wine.shape)


#describing data
wine.describe(include='all')


# Finding Null Values

print(wine.isna().sum())
wine.corr()
wine.groupby('quality').mean()




# Distplot:

wine.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)



# Heatmap for expressing correlation

corr = wine.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True, cmap='coolwarm')




# Feature Selection
# Create Classification version of target variable
wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]# Separate feature variables and target variable
X = wine.drop(['quality','goodquality'], axis = 1)
Y = wine['goodquality']


#plot of each feature in each wine
X.plot(style='.',
        figsize=(15, 5),
        color=[color_pal[0],color_pal[1],color_pal[2],color_pal[3],color_pal[4],color_pal[5],color_pal[6],color_pal[7],color_pal[8],color_pal[9]],
        title='Amount of each by wine')
plt.show()


#plot of each wine as good or bad
Y.plot(style='.',
        figsize=(15, 5),
        color=[color_pal[0]],
        title='GOOD/BAD WINES')
plt.show()

# Splitting Dataset

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)


# LogisticRegression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("LRAccuracy Score:",accuracy_score(Y_test,Y_pred))


# Using KNN:

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("KNNAccuracy Score:",accuracy_score(Y_test,y_pred))

# Using SVC

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,Y_train)
pred_y = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("SVCAccuracy Score:",accuracy_score(Y_test,pred_y))

# Using Decision Tree:

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("DTAccuracy Score:",accuracy_score(Y_test,y_pred))

# Using GaussianNB:

from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("GNBAccuracy Score:",accuracy_score(Y_test,y_pred3))

# Using Random Forest:

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1,n_estimators=1000)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
res = format(accuracy_score(Y_test,y_pred2),'.8f')
print("RANFORAccuracy Score:",res)


#using XGBOOST
import xgboost as xgb
model5 = xgb.XGBClassifier(random_state=1)
model5.fit(X_train, Y_train)
y_pred5 = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("XGAccuracy Score:",accuracy_score(Y_test,y_pred5))




#random input check- finalising model2 (RandomForest model) as our final model
x=[[8.0,0.9,0.01,2.4,0.082,15,50,0.9873,3.8,5,50]]
ans=model2.predict(x)
if ans==[0]:
  print('Bad wine')
else:
  print('Good wine')



#feature importance extraction
imp=model2.feature_importances_
columns=X.columns
i=0

while i<len(columns):
  print("The imp of feature ",columns[i],' is ' ,round((imp[i] * 100),2),'%.')
  i+=1



#accuracy after oversampling
from imblearn.over_sampling import SMOTE
os= SMOTE(k_neighbors=4)
X,Y = os.fit_resample(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)

model2 = RandomForestClassifier(random_state=7,n_estimators=1000)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)
res1=format(accuracy_score(Y_test,y_pred2),'.8f')



#finding the best parameters using Random search with cross validation

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf = RandomForestClassifier(random_state = 42)
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 10, scoring='neg_mean_absolute_error',
                              cv = 3, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)

rf_random.fit(X_train, Y_train)

rf_random.best_params_

best_random = rf_random.best_estimator_
predictions = best_random.predict(X_test)
res2=format(accuracy_score(Y_test,predictions),'.8f')




#dropping less correlated features using matrix
X = wine.drop(['quality','goodquality','pH'], axis = 1)
Y = wine['goodquality']
os= SMOTE(k_neighbors=4)
X,Y = os.fit_resample(X,Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)

best_random.fit(X_train, Y_train)
y_pred2 = best_random.predict(X_test)
res3=format(accuracy_score(Y_test,y_pred2),'.8f')


#Finding CROSS-VALIDATION score

from sklearn.model_selection import cross_val_score

res4=np.mean(cross_val_score(model2,X,Y,cv=7))
res5=format(res4,'.8f')



#plotting all scores
l=[]
l.append(res)
l.append(res1)
l.append(res2)
l.append(res3)
l.append(res5)
l=sorted(l)
hehe=['BaseAcc','OverSample','HyperTunin','DropFeature','CrossValScor']
barplot=plt.bar(x=hehe,height=l,fc='lightgray',ec='black')
plt.bar_label(barplot,labels=l,label_type='edge')
plt.show()