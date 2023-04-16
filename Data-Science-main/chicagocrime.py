import pandas as pd

df = pd.read_csv('D:\PycharmProjects\chicago_train.csv')

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr = LogisticRegression(random_state=0)
rfc = RandomForestClassifier(random_state=1)
dtc = DecisionTreeClassifier(random_state=0)
svm = svm.SVC()
mlp = MLPClassifier(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(5,2), random_state=0)
gbc = GradientBoostingClassifier(n_estimators=10)

df.drop(['sr','ID','Description','Location Description','Date','Block','Location','FBI Code','Case Number','X Coordinate','Y Coordinate','Community Area','Updated On','IUCR'], inplace=True, axis=1)


nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df.dropna(subset = ['Ward','District','Latitude','Longitude'], inplace=True)
df.drop_duplicates()

le = LabelEncoder()
df['Arrest'] = le.fit_transform(df['Arrest'])
df['Primary Type'] = le.fit_transform(df['Primary Type'])

x = df.drop('Arrest',axis=1)
y = df['Arrest']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3 )

logr.fit(x_train, y_train)
y_predict1 = logr.predict(x_test)
print('Logistic', accuracy_score(y_test, y_predict1))

rfc.fit(x_train, y_train)
y_predict2 = rfc.predict(x_test)
print('Random Forest', accuracy_score(y_test, y_predict2))

dtc.fit(x_train, y_train)
y_predict3 = dtc.predict(x_test)
print('Decision Tree', accuracy_score(y_test, y_predict3))

svm.fit(x_train, y_train)
y_predict4 = svm.predict(x_test)
print('Support Vector', accuracy_score(y_test, y_predict4))

mlp.fit(x_train, y_train)
y_predict5 = mlp.predict(x_test)
print('MLP', accuracy_score(y_test,  y_predict5))

gbc.fit(x_train, y_train)
y_predict6 = gbc.predict(x_test)
print('Gradient Boosting', accuracy_score(y_test,  y_predict6))

#OUTPUT

# Logistic 0.7448591012947449
# Random Forest 0.8347296268088348
# Decision Tree 0.7928408225437928
# Support Vector 0.7448591012947449
# MLP 0.7448591012947449
# Gradient Boosting 0.853008377760853
