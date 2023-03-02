# A. Doc du lieu wine
#Doc bang thu vien pandas
import pandas as pd
dulieu = pd.read_csv("winequality-red.csv",delimiter=';')
x = dulieu.iloc[:,0:11]
y= dulieu.quality
print("Cau A: Nhan du lieu tu winequality-red.csv")
print(dulieu)
print("--------------------------------------------------------------")
#B thong tin cua du lieu
print("Cau B: Thong tin cua du lieu: \n")
print("So luong phan tu: ",len(x))
import numpy as np
print("Cac gia tri khac nhau cua bien: ",np.unique(y))
print("So luong va gia tri khac nhau cua bien:\n ",y.value_counts())

print("--------------------------------------------------------------")
print("Cau C: Phan chia du lieu thanh 2 phan: \n")
#C: Phan chia du lieu thanh tap test va train (mo hinh Gaussian)
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4,random_state=0)
print("So luong phan tu trong tap test: ",len(x_test))
print("Cac phan tu cua nhan trong tap test: ",np.unique(y_test))

print("--------------------------------------------------------------")
print("Cau D: Xay dung mo hinh KNN: \n")
#D: Mo hinh KNN
from sklearn.neighbors import KNeighborsClassifier
Mohinh_KNN = KNeighborsClassifier(n_neighbors=7)
Mohinh_KNN.fit(x_train,y_train)

print("--------------------------------------------------------------")
print("Cau D1-2: Tinh do chinh xac cua mo hinh KNN: \n")
#Tinh do chinh xac mo hinh KNN
from sklearn.metrics import accuracy_score
y_pred = Mohinh_KNN.predict(x_test)
print("Do chinh xac tong the cua KNN voi tat ca cac phan tu: ", accuracy_score(y_test,y_pred)*100)
from sklearn.metrics import confusion_matrix
m1 = confusion_matrix(y_test,y_pred)
print("Danh gia mo hinh: \n",m1)

print("Do chinh xac cua lop 3: ",0 if m1[0,0]==0 else m1[0,0]/(m1[0,0]+m1[0,1]+m1[0,2]+m1[0,3]+m1[0,4]+m1[0,5]))
print("Do chinh xac cua lop 4: ",0 if m1[1,1]==0 else m1[1,1]/(m1[1,0]+m1[1,1]+m1[1,2]+m1[1,3]+m1[1,4]+m1[1,5]))
print("Do chinh xac cua lop 5: ",0 if m1[2,2]==0 else m1[2,2]/(m1[2,0]+m1[2,1]+m1[2,2]+m1[2,3]+m1[2,4]+m1[2,5]))
print("Do chinh xac cua lop 6: ",0 if m1[3,3]==0 else m1[3,3]/(m1[3,0]+m1[3,1]+m1[3,2]+m1[3,3]+m1[3,4]+m1[3,5]))
print("Do chinh xac cua lop 7: ",0 if m1[4,4]==0 else m1[4,4]/(m1[4,0]+m1[4,1]+m1[4,2]+m1[4,3]+m1[4,4]+m1[4,5]))
print("Do chinh xac cua lop 8: ",0 if m1[5,5]==0 else m1[5,5]/(m1[5,0]+m1[5,1]+m1[5,2]+m1[5,3]+m1[5,4]+m1[5,5]))


y_test8 = y_test.iloc[1:9]
x_test8 = x_test.iloc[1:9]
y_pred8 = Mohinh_KNN.predict(x_test8)
print("\nDu doan voi 8 phan tu dau tien: \n Cac gia tri khac nhau cua nhan khi du lieu la 8: ",np.unique(y_test8))
print("Do chinh xac cua KNN voi 8 cac phan tu: ", accuracy_score(y_test8,y_pred8)*100)
m2 = confusion_matrix(y_test8,y_pred8)
print("Danh gia mo hinh: \n",m2)
print("Do chinh xac cua lop 4: ",0 if m2[0,0]==0 else m2[0,0]/(m2[0,0]+m2[1,0]+m2[2,0]+m2[3,0]))
print("Do chinh xac cua lop 5: ",0 if m2[1,1]==0 else m2[1,1]/(m2[0,1]+m2[1,1]+m2[2,1]+m2[3,1]))
print("Do chinh xac cua lop 6: ",0 if m2[2,2]==0 else m2[2,2]/(m2[0,2]+m2[1,2]+m2[2,2]+m2[3,2]))
print("Do chinh xac cua lop 7: ",0 if m2[3,3]==0 else m2[3,3]/(m2[0,3]+m2[1,3]+m2[2,3]+m2[3,3]))
print("--------------------------------------------------------------")
print("Cau E: Xay dung mo hinh Bayes va danh gia: \n")
#E: xay dung mo hinh bayes
model = GaussianNB()
model.fit(x_train,y_train)

thucte= y_test
dubao = model.predict(x_test)
thucte
dubao
print("Do chinh xac tong the cua Bayes voi tat ca cac phan tu: ", accuracy_score(thucte,dubao)*100)
m1 = confusion_matrix(thucte,dubao)
print("Danh gia mo hinh: \n",m1)
print("Do chinh xac cua lop 3: ",m1[0,0]/(m1[0,0]+m1[0,1]+m1[0,2]+m1[0,3]+m1[0,4]+m1[0,5]))
print("Do chinh xac cua lop 4: ",m1[1,1]/(m1[1,0]+m1[1,1]+m1[1,2]+m1[1,3]+m1[1,4]+m1[1,5]))
print("Do chinh xac cua lop 5: ",m1[2,2]/(m1[2,0]+m1[2,1]+m1[2,2]+m1[2,3]+m1[2,4]+m1[2,5]))
print("Do chinh xac cua lop 6: ",m1[3,3]/(m1[3,0]+m1[3,1]+m1[3,2]+m1[3,3]+m1[4,4]+m1[3,5]))
print("Do chinh xac cua lop 7: ",m1[4,4]/(m1[4,0]+m1[4,1]+m1[4,2]+m1[4,3]+m1[4,4]+m1[4,5]))
print("Do chinh xac cua lop 8: ",m1[5,5]/(m1[5,0]+m1[5,1]+m1[5,2]+m1[5,3]+m1[5,4]+m1[5,5]))

print("--------------------------------------------------------------")
print("Cau F: Danh gia 2 mo hinh Bayes va KNN : \n")
#F: Hold on voi 2/3 hoc 1/3 kiem tra va do sanh do chinh xac KNN vs Bayes
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3,random_state=0)

#Xay dung mo hinh KNN
Mohinh_KNN = KNeighborsClassifier(n_neighbors=7)
Mohinh_KNN.fit(x_train,y_train)

#tinh do chinh xac tong the cua KNN
y_pred = Mohinh_KNN.predict(x_test)
print("Do chinh xac tong the cua KNN voi tat ca cac phan tu: ", accuracy_score(y_test,y_pred)*100)
KS = accuracy_score(y_test,y_pred)*100

#Xay dung mo hinh Bayes
model = GaussianNB()
model.fit(x_train,y_train)
thucte= y_test
dubao = model.predict(x_test)
thucte
dubao
print("Do chinh xac tong the cua Bayes voi tat ca cac phan tu: ", accuracy_score(thucte,dubao)*100)
BS = accuracy_score(thucte,dubao)*100
if KS>BS :
	print("Do chinh xac cua KNN cao hon Bayes")
else:
	print("Do chinh xac cua KNN thap hon Bayes")
