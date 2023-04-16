import pandas as pd
df=pd.read_csv('D:\PycharmProjects\census_income.csv')

df=df.replace('?','')

#Label Encoding
from sklearn.preprocessing import LabelEncoder
for col in df.columns.values:
    if df[col].dtype=='object':
        le=LabelEncoder()
        df[col]=le.fit_transform(df[col])

#feature selection
x=df.drop('income',axis=1)
y=df['income']


x=x.drop(['workclass','education','race','sex','capital.loss','native.country'], axis=1)

#Oversampling as there is imbalance in data
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
ros.fit(x,y)
x,y=ros.fit_resample(x,y)

#Model training/testing and results
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import mean_squared_error

dt=DecisionTreeClassifier(random_state=0)
rf=RandomForestClassifier(random_state=0)
gb=GradientBoostingClassifier(n_estimators=10)
lr=LogisticRegression(random_state=0)
sc=svm.SVC(random_state=0)
gnb=GaussianNB()
mnb=MultinomialNB()
bn=BernoulliNB()

x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, train_size=0.3)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print('Decision tree',accuracy_score(y_test,y_pred))

rf.fit(x_train,y_train)
y_pred1=rf.predict(x_test)
print('Random forest',accuracy_score(y_test,y_pred1))

gb.fit(x_train,y_train)
y_pred2=gb.predict(x_test)
print('Gradient Boosting',accuracy_score(y_test,y_pred2))

lr.fit(x_train,y_train)
y_pred3=lr.predict(x_test)
print('Logistic regression',accuracy_score(y_test,y_pred3))

sc.fit(x_train,y_train)
y_pred4=sc.predict(x_test)
print('SVM',accuracy_score(y_test,y_pred4))

gnb.fit(x_train,y_train)
y_pred5=gnb.predict(x_test)
print('Guassian naive bayes',accuracy_score(y_test,y_pred5))

bn.fit(x_train,y_train)
pred=bn.predict(x_test)
print('Bernoullis Naive bayes',accuracy_score(y_test,pred))

#OUTPUT
'''
Decision tree 0.8389100785945446
Random forest 0.871908229311142
Gradient Boosting 0.8041204345815997
Logistic regression 0.6027797041146555
SVM 0.5898636153490523
Guassian naive bayes 0.5971162736939436
Bernoullis Naive bayes 0.7430362921867776
'''

#MEAN SQUARED ERROR
print('Decision tree',mean_squared_error(y_test,y_pred))
print('Random forest',mean_squared_error(y_test,y_pred1))
print('Gradient Boosting',mean_squared_error(y_test,y_pred2))
print('Logistic regression',mean_squared_error(y_test,y_pred3))
print('SVM',mean_squared_error(y_test,y_pred4))
print('Guassian naive bayes',mean_squared_error(y_test,y_pred5))
print('Bernoullis Naive bayes',mean_squared_error(y_test,pred))


#OUTPUT
'''
Decision tree 0.1610899214054554
Random forest 0.12809177068885808
Gradient Boosting 0.19587956541840038
Logistic regression 0.3972202958853444
SVM 0.41013638465094776
Guassian naive bayes 0.4028837263060564
Bernoullis Naive bayes 0.25696370781322236
'''