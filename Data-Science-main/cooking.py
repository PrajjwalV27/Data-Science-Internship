import pandas as pd

df=pd.read_json('D:/PDSML/train.json')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns


rf=RandomForestClassifier(random_state=1)
nb=MultinomialNB()
dt=DecisionTreeClassifier(random_state=0)

d_c=['greek','southern_us' ,'filipino' ,'indian' ,'jamaican', 'spanish', 'italian','mexican', 'chinese' ,'british' ,'thai' ,'vietnamese' ,'cajun_creole','brazilian' ,'french' ,'japanese' ,'irish' ,'korean' ,'moroccan' ,'russian']

print(df['cuisine'].unique())
x=df['ingredients']
y=df['cuisine'].apply(d_c.index)

df['all_ingredients']=df['ingredients'].map(";".join)

cv=CountVectorizer()
x=cv.fit_transform(df['all_ingredients'].values)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)

print(accuracy_score(y_test,y_pred))



#OUTPUT

# ['greek' 'southern_us' 'filipino' 'indian' 'jamaican' 'spanish' 'italian'
#  'mexican' 'chinese' 'british' 'thai' 'vietnamese' 'cajun_creole'
#  'brazilian' 'french' 'japanese' 'irish' 'korean' 'moroccan' 'russian']
# 0.7583909490886235