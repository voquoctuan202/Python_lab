#KNN

from sklearn.datasets import load_iris
iris_dt = load_iris()

iris_dt.data[1:5]
iris_dt.target[1:5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dt.data,iris_dt.target, test_size=1/3.0,random_state=5)

X_train[1:6]
X_train[1:6,1:3]
y_train[1:6]
X_test[6:10]
y_test[6:10]

#Mo hinh KNN
from sklearn.neighbors import KNeighborsClassifier
Mohinh_KNN = KNeighborsClassifier(n_neighbors=5)
Mohinh_KNN.fit(X_train,y_train)

#Du doan
y_pred = Mohinh_KNN.predict(X_test)
y_test
Mohinh_KNN.predict([[4,4,3,3]])

#Tinh do chinh xac
from sklearn.metrics import accuracy_score
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred, labels=[2,0,1]))

#Bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

#Nhan du lieu bang pandas csv
import pandas as pd
dulieu = pd.read_csv("iris_data.csv")
x = dulieu.iloc[:,0:4]
y= dulieu.nhan
#Phan chia du lieu thanh tap test va train (mo hinh Gaussian)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=0)
model = GaussianNB()
model.fit(x_train,y_train)
print(model)

#du doan
thucte= y_test
dubao = model.predict(x_test)
thucte
dubao

#danh gia gia thuat
from sklearn.metrics import confusion_matrix
cnf_matrix_gnb = confusion_matrix(thucte,dubao)
print(cnf_matrix_gnb)
