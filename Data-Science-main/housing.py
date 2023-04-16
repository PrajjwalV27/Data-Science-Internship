import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm


column_names = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']
df=pd.read_csv('D:\PycharmProjects\Housing.csv')
df['chas'].fillna(0,inplace=True)

for col in column_names:
    if df[col].count()!=506:
        df[col].fillna(df[col].median(),inplace=True)

x=df.drop('medv',axis=1)
y=df['medv']


lr=LinearRegression()
rf=RandomForestRegressor()
gb=GradientBoostingRegressor()
sv=svm.SVR()

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print('Linear Regression',mean_squared_error(y_test,y_pred))

rf.fit(x_train,y_train)
y_pred1=rf.predict(x_test)
print('Random Forest',mean_squared_error(y_test,y_pred1))

gb.fit(x_train,y_train)
y_pred2=gb.predict(x_test)
print('Gradient Boosting',mean_squared_error(y_test,y_pred2))

sv.fit(x_train,y_train)
y_pred4=sv.predict(x_test)
print('SVM',mean_squared_error(y_test,y_pred4))

#OUTPUT
'''
Linear Regression 27.195965766883294
Random Forest 15.468033243421054
Gradient Boosting 13.345947571216808
SVM 68.18381202764185
'''

