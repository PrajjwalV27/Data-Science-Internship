import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

logr = LinearRegression()
gb = GradientBoostingRegressor(n_estimators=100)
dt =  DecisionTreeRegressor(max_depth=5)

df=pd.read_csv('D:\PycharmProjects\walmart_train.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['days']=df['Date'].dt.day
df['month']=df['Date'].dt.month
df['year']=df['Date'].dt.year
df['WeekOfYear'] = df.Date.dt.isocalendar().week
df.drop('Date',axis=1,inplace=True)

le = LabelEncoder()
df['IsHoliday'] = le.fit_transform(df['IsHoliday'])


x=df.drop('Weekly_Sales',axis=1)
y=df['Weekly_Sales']

#Feature Extraction
pca = PCA(n_components=3)
fit = pca.fit(x)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.3)

logr.fit(X_Train, Y_Train)
gb.fit(X_Train, Y_Train)
dt.fit(X_Train, Y_Train)

print('Linear Regression',logr.score(X_Test,Y_Test))
print('Gradient Boosting',gb.score(X_Test, Y_Test))
print('Decision Tree',dt.score(X_Test, Y_Test))

#OUTPUT
# Linear Regression 0.029047991527707873
# Gradient Boosting 0.6257456244594666
# Decision Tree 0.49118525067358154
























