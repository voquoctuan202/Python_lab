#Doc du lieu
from sklearn.datasets import load_iris
iris_dt = load_iris()

iris_dt.data[1:5]
iris_dt.target[1:5]
#Nghi thuc Hold_out
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dt.data,iris_dt.target, test_size=1/3.0,random_state=5)

X_train[1:6]
X_train[1:6,1:3]
y_train[1:6]
X_test[6:10]
y_test[6:10]

from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state =100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train,y_train)

y_pred = clf_gini.predict(X_test)
clf_gini.predict([[4,4,3,3]])

from sklearn.metrics import accuracy_score
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred, labels=[2,0,1]))

#Nghi thuc K-fold
from sklearn.model_selection import KFold
kf= KFold(n_splits=15) # chia tập dữ liệu thành 15 phần

X= iris_dt.data
y= iris_dt.target
for train_index, test_index in kf.split(X,y):
    print("Train: ",train_index,"Test: ",test_index)
    # X_train,x_test = X[train_index], X[test_index]
    # y_train,y_test = y[train_index], y[test_index]
    print(X_train)
    clf_gini.fit(X[train_index],y[train_index])
    y_pred = clf_gini.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
    # print("X_test",X_test)
    print("=======================")

#Bai toan cay hoi quy
import pandas as pd
dulieu = pd.read_csv("housing_RT.csv",index_col=0,delimiter=",")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dulieu.iloc[:,1:5],dulieu.iloc[:,0], test_size=1/3.0,random_state=5)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
err = mean_squared_error(y_test,y_pred)
import numpy as np
print(np.sqrt(err))
