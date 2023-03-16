import numpy as np
import matplotlib.pyplot as plt

#1.Vi du du doan gia nha
x = np.array([1,2,4])
y = np.array([2,3,6])

#Ve do thi 
plt.axis([0,5,0,8])
# plt.plot(x,y,'ro',color="blue")
plt.xlabel("Gia tri thuoc tinh X")
plt.ylabel("Gia tri du doan Y")
# plt.show()

#cong thuc tinh theta
def LR1(x,y,eta,lanlap,theta0 , theta1):
    m = len(x)
    for k in range(0,lanlap):
        print("Lan lap: ",k)
        for i in range(0,m):
            h_i=theta0+theta1*x[i]
            #theta0
            theta0 = theta0 + eta*(y[i]-h_i)*1
            print("Phan tu ", i,"y=",y[i], "h=",h_i,"gia tri theta0 = ",round(theta0,3))
            #theta1
            theta1=   theta1+ eta*(y[i]-h_i)*x[i]
            print ("Phan tu ", i, "gia tri cua theta1 =",round(theta1,3))
    return [round(theta0,3),round(theta1,3)] 



#Ve duong hoi quy
theta = LR1(x,y,0.2,1,0,1)
print("Theta",theta)
X1= np.array([1,6])
Y1= theta[0] + theta[1]*X1
print("Y1: ",Y1, "\nX1",X1)
theta2 = LR1(x,y,0.2,2,0,1)
X2= np.array([1,6])
Y2= theta2[0] + theta2[1]*X2

plt.axis([0,7,0,10])
plt.plot(x,y,'ro',color="blue")

plt.plot(X1,Y1,color="violet")
plt.plot(X2,Y2,color="green")

plt.xlabel("Gia tri thuoc tinh X")
plt.ylabel("Gua tri du doan Y")
plt.show()

#Du doan cho phan tu moi
XX = [0,3,5]
for i in range(0,3):
    YY = theta[0] + theta[1]*XX[i]
    print("X=",XX[i]," => y=",round(YY,3))


# #2. Sử dụng thư viện scikit-learn của Python để tìm các giá trị theta

# #Doc du lieu
# import pandas as pd
# dt = pd.read_csv("Housing_2019.csv",index_col=0)
# dt.iloc[2:4:]
# X= dt.iloc[:,[1,2,3,4,10]]
# X.iloc[1:5,]
# Y=dt.price
# print(X)

# # plt.scatter(dt.lotsize, dt.price)
# # plt.show()

# #Huan luyen mo hinh
# import sklearn
# from sklearn import linear_model
# lm =  linear_model.LinearRegression()
# lm.fit(X[0:520],Y[0:520])

# print (lm.intercept_)
# print (lm.coef_)

# #Du bao gia nha cho 20 phan tu cuoi cung
# y = dt.price
# y_test = y[-20:]
# X_test = X[-20:]
# y_pred = lm.predict(X_test)

# #so sanh voi gia tri thuc te va gia tri du bao
# print(y_pred)
# print(y_test)

# from sklearn.metrics import mean_squared_error
# err = mean_squared_error(y_test,y_pred)
# print("MSE: ",round(err,3))
# rmse_ree= np.sqrt(err)
# print("RMSE",round(rmse_ree,3))

# #B: PHƯƠNG PHÁP TẬP HỢP MÔ HÌNH
# #Su dung giai thuat Bagging tren tap du lieu du doan gia nha

# #Du doan gia nha bang cay quyet dinh
# import pandas as pd 
# from sklearn.metrics import mean_squared_error
# import numpy as np

# dt = pd.read_csv("Housing_2019.csv",index_col=0)
# X=dt.iloc[:,[1,2,3,4,10]]
# Y = dt.price

# import sklearn 
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test= train_test_split(X,Y,test_size=1/3,random_state=100)
# len(X_train)

# from sklearn.ensemble import BaggingRegressor
# from sklearn.tree import DecisionTreeRegressor
# tree = DecisionTreeRegressor(random_state=0)

# bagging_regtree = BaggingRegressor(base_estimator=tree, n_estimators=10, random_state=42)
# bagging_regtree.fit(X_train,y_train)
# y_pred = bagging_regtree.predict(X_test)
# err = mean_squared_error(y_test, y_pred)
# # print("MSE= ",err)
# # print("RMSE= ",np.sqrt(err))

# #Su dung giai thuat hoi quy tuyen tinh

# import pandas as pd 
# from sklearn.metrics import mean_squared_error
# import numpy as np

# dt = pd.read_csv("Housing_2019.csv",index_col=0)
# X=dt.iloc[:,[1,2,3,4,10]]
# Y = dt.price

# import sklearn 
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test= train_test_split(X,Y,test_size=1/3,random_state=100)
# len(X_train)

# from sklearn.ensemble import BaggingRegressor
# from sklearn import linear_model
# lm = linear_model.LinearRegression()
# bagging_reg = BaggingRegressor(base_estimator=lm, n_estimators=10, random_state=42)
# bagging_reg.fit(X_train,y_train)
# y_pred= bagging_reg.predict(X_test)
# err= mean_squared_error(y_test,y_pred)
# print("MSE= ",err)
# print("RMSE= ",np.sqrt(err))
