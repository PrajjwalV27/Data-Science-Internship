import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv("D:/PycharmProjects/train.csv")
test=pd.read_csv('D:/PycharmProjects/test.csv')


df['source']='train'
test['source']='test'
data=pd.concat([df,test])

#Dropping Unwanted Features
data.drop('Product_Category_3', axis = 1, inplace = True)
data.drop('User_ID', axis = 1, inplace = True)
data.drop('Product_ID', axis = 1, inplace = True)

#Filling Missing Values
data['Product_Category_2'].fillna(data['Product_Category_2'].mean(), inplace = True)

#Replacing
data['Age']=data['Age'].apply(lambda x: str(x).replace('55+','55'))
data['Stay_In_Current_City_Years']=data['Stay_In_Current_City_Years'].apply(lambda x: str(x).replace('4+','4'))

#Label Encoding
le=LabelEncoder()
data['Gender']=le.fit_transform(data['Gender'])
le=LabelEncoder()
data['Age']=le.fit_transform(data['Age'])
le=LabelEncoder()
data[ 'City_Category']=le.fit_transform(data['City_Category'])

#Converting Datatype to int
data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].astype('int')

#Separating Train and Test
train=data.loc[data['source']=='train']
test=data.loc[data['source']=='test']
train.drop('source', axis = 1, inplace = True)
test.drop('source', axis = 1, inplace = True)

X = train.drop("Purchase", axis = 1)
Y = train["Purchase"]

selector = ExtraTreesRegressor()
selector.fit(X, Y)
feature_imp = selector.feature_importances_
X.drop(['Gender', 'City_Category', 'Marital_Status'], axis = 1, inplace = True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
lin_reg = LinearRegression()
knn = KNeighborsRegressor()
dec_tree = DecisionTreeRegressor()
ran_for = RandomForestRegressor()
dtc=DecisionTreeClassifier()

print("MEAN SQUARED ERRORS")

lin_reg.fit(X_train, Y_train)
Y_pred_lin_reg = lin_reg.predict(X_test)
print("Linear Regression: ",mean_squared_error(Y_test, Y_pred_lin_reg))


knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
print("KNN regression: ",mean_squared_error(Y_test, Y_pred_knn))


dec_tree.fit(X_train, Y_train)
Y_pred_dec = dec_tree.predict(X_test)
print("Decision tree regression: ",mean_squared_error(Y_test, Y_pred_dec))


ran_for.fit(X_train, Y_train)
Y_pred_ran_for = ran_for.predict(X_test)
print("Random forest regression: ",mean_squared_error(Y_test, Y_pred_ran_for))


'''
MEAN SQUARED ERRORS
Linear Regression:  22044840.101023477
KNN regression:  10469708.932551857
Decision tree regression:  9843766.502839973
Random forest regression:  9058790.950768305
'''