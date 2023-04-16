import pandas as pd
df=pd.read_csv('D:/PycharmProjects/tested.csv')
from sklearn.ensemble import ExtraTreesClassifier

df=df.drop('PassengerId',axis=1)
df=df.drop('Cabin',axis=1)
df=df.drop('Name',axis=1)
df=df.drop('Ticket',axis=1)

df['Fare'].fillna(df['Fare'].mean(),inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Sex'])
df['Sex']=le.transform(df['Sex'])
le = LabelEncoder()
le.fit(df['Embarked'])
df['Embarked']=le.transform(df['Embarked'])
df['Age'].fillna(df['Age'].median(),inplace=True)
df1=df.drop('Survived',axis=1)
et = ExtraTreesClassifier()
et.fit(df1,df['Survived'])

df=df.drop('Age',axis=1)
df=df.drop('Embarked',axis=1)
df=df.drop('Pclass',axis=1)

x=df.drop('Survived',axis=1)
y=df['Survived']
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2,k='all')
bestfeatures.fit(x,y)
dfscore=pd.DataFrame(bestfeatures.scores_)
dfcolumns=pd.DataFrame(x.columns)
featuresScores=pd.concat([dfcolumns,dfscore],axis=1)
featuresScores.columns=['Features','Score']

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
df['SibSp']=pd.cut(df['SibSp'],2,labels=[0,1])

sc=svm.SVC(random_state=0)
mnb=MultinomialNB()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, train_size=0.3)

sc.fit(x_train,y_train)
y_pred1=sc.predict(x_test)
print('SVM',accuracy_score(y_test,y_pred1))

mnb.fit(x_train,y_train)
y_pred2=mnb.predict(x_test)
print('MultinomialNB',accuracy_score(y_test,y_pred2))

#OUTPUT
# SVM 0.6279863481228669
# MultinomialNB 0.825938566552901

