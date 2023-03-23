import numpy as np
import matplotlib.pyplot as plt

#Câu 1: Nhập 2 mảng
x = np.array([147,150,153,158, 163, 165, 168, 170, 173, 175, 178, 180, 183])
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

#Vẽ đồ thị biểu diễn tập dữ liệu
plt.axis([0,185,0,95])
plt.title("Biểu đồ thể hiện tập dữ liệu")
plt.plot(x,y,'o',color="blue")
plt.xlabel("Gia tri thuoc tinh X")
plt.ylabel("Gia tri du doan Y")
plt.show()

#xây dựng hàm tính theta
def LR1(x,y,eta,lanlap,theta0,theta1):
    m = len(x)
    for k in range(0,lanlap):
        print("Lan lap: ",k)
        for i in range(0,m):
            h_i= theta0 + theta1*x[i]
            #theta0
            theta0 = theta0 + eta*(y[i]-h_i)*1
            print("Phan tu ", i,"y=",y[i], "h=",h_i,"gia tri theta0 = ",round(theta0,3))
            #theta1
            theta1=   theta1+ eta*(y[i]-h_i)*x[i]
            print ("Phan tu ", i, "gia tri cua theta1 =",round(theta1,3))
    return [round(theta0,3),round(theta1,3)]



#Vẽ đường hồi quy: cần phải giảm tốc độ học
theta = LR1(x,y,0.00002,3,0,1)
X1= np.array([1,200])
Y1= theta[0] + theta[1]*X1

plt.axis([0,185,0,95])
plt.plot(x,y,'o',color="blue")

print("X1: ",X1,"Y1: ",Y1)
plt.title("Biểu đồ thể hiện đường hồi quy trên tập dữ liệu")
plt.plot(X1,Y1,color="violet")


plt.xlabel("Gia tri thuoc tinh X")
plt.ylabel("Gua tri du doan Y")
plt.show()
#========================================================================
#Câu 2:
#Doc du lieu
import pandas as pd
dt = pd.read_csv("Housing_2019.csv",index_col=0)
dt.iloc[2:4:]
X= dt.iloc[:,[1,2,4,10]]
X.iloc[1:5,]
Y=dt.price
print(X)

#Nghi thức Hold_out
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dt.iloc[:,1:5],dt.iloc[:,0], test_size=1/3.0,random_state=5)

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn import linear_model
lm = linear_model.LinearRegression()
bagging_reg = BaggingRegressor(base_estimator=lm, n_estimators=10, random_state=42)
bagging_reg.fit(X_train,y_train)
y_pred= bagging_reg.predict(X_test)
err= mean_squared_error(y_test,y_pred)

#Đánh giá với 2 chỉ số MSE và RMSE
print("MSE= ",err)
print("RMSE= ",np.sqrt(err))



#Câu 3:
#Nhập dữ liệu từ file vatlieu.csv
data = pd.read_csv("vatlieu.csv",index_col=0)

#Tách dữ liệu gỗ cứng, và độ căng ra 2 mảng X,Y
X =list(data.Go_cung)
Y= list(data.Do_cang)

#Biểu đồ thể hiện mối liên hệ
plt.axis([0,30,0,60])
plt.title("Biểu đồ thể hiện mối liên hệ của gỗ cứng và độ căng")
plt.plot(X,Y,color="violet")
plt.show()

#Công thức liên hệ
theta = LR1(X,Y,0.01,1,0,1)
print("Phương trình thể hiện mối liên hệ: H0(X) = ",round(float(theta[0]),3), "+",round(float(theta[1]),3),"X1")
X1= np.array([1,6])
print("Phương trình thể hiện mối liên hệ:")
Y1= theta[0] + theta[1]*X1


#Câu 4: Nhập dữ liệu rượu vang.
import pandas as pd
dt = pd.read_csv("winequality-red.csv",delimiter=";")
X= dt.iloc[:,0:11]
Y=dt.quality

#Nghi thức Hold_out
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=1/3.0,random_state=5)

#Xây dựng mô hình rừng ngẫu nhiên (RandomForestClassifier)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50)
#Huấn luyện
classifier.fit(X_train, y_train)
#Test
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
#Report mô hình
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
#Độ chính xác tổng thể
print("Accuracy:", round(result2,3))

#Cau 5: Nhập dữ liệu rượu vang
import pandas as pd
dt = pd.read_csv("winequality-red.csv",delimiter=";")
X= dt.iloc[:,0:11]
Y= dt.quality

#Nghi thức K-fold
Fold_n=5
from sklearn.model_selection import KFold
kf= KFold(n_splits=Fold_n,shuffle=True)

#Xây dựng mô hình AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
# Import Support Vector Classifier
from sklearn.svm import SVC
#Import scikit-learn metrics module for accuracy calculation
svc=SVC(probability=True, kernel='linear')

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)

Fold_index =0 # Dùng để đếm số lần duyệt
Sum_AS=0 #Tính tổng độ chính xác
for train_index, test_index in kf.split(X,Y):
    print("Lan duyet thu ",Fold_index+1)

    abc.fit(X.iloc[train_index], Y.iloc[train_index])
    y_pred = abc.predict(X.iloc[test_index])

    result = confusion_matrix(Y.iloc[test_index], y_pred)
    print("Confusion Matrix:")
    print(result)
    # result1 = classification_report(Y.iloc[test_index], y_pred)
    # print("Classification Report:",)
    # print (result1)
    result2 = accuracy_score(Y.iloc[test_index],y_pred)
    print("Accuracy:", round(result2,3))
    Sum_AS = Sum_AS + result2*100

    Fold_index = Fold_index+1
    print("=================================================================")
print("Accuracy AVG:", round(Sum_AS/Fold_n,3))
