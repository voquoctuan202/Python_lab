print("=========================================================================")
print("=============================   Bài 1   =================================")
print("=========================================================================")
#Bài 1:
#A.Đọc dữ liệu
import pandas as pd
dulieu = pd.read_csv("winequality-white.csv",delimiter=";")

print(dulieu)
print("Du lieu trong file có",len(dulieu.iloc[0])-1,"thuộc tính \nCột",dulieu['quality'].name,"là cột nhãn")
import numpy as np
a = np.unique(dulieu['quality'])
print("Gia trị của nhãn", a)
#B.Du liệu trên có 12 cột trong đó 11 cột đầu là thuộc tính cột cuối là cột nhãn

#C. Thực hiện nghi thức K_Fold
Fold_n=50
from sklearn.model_selection import KFold
kf= KFold(n_splits=Fold_n,shuffle=True) # chia tập dữ liệu thành Fold_n phần và có xáo trộn

#D. Xây dụng mô hình cây quyết định
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state =100, max_depth=3, min_samples_leaf=5)

#import thư viện tính độ chính xác tổng thể
from sklearn.metrics import accuracy_score

#Phân chia dữ liệu thành 2 phần
X= dulieu.iloc[:,0:10] # X là 11 cột thuộc tính
y= dulieu.iloc[:,11] # y là cột nhãn
# print(X)
# print(y)
Fold_index =0 # Dùng để đếm số lần duyệt
Sum_AS=0 #Tính tổng độ chính xác của 50 lần duyệt
for train_index, test_index in kf.split(X,y):
    print("Lan duyet thu ",Fold_index+1)

    #In số lượng phân tử train và test
    print("Số lượng phần tử Train: ",len(train_index),"\nSố lượng phần tử Test: ",len(test_index))
    #Sô lượng của tập train là 4800 hoặc 4801
    #Số lượng của tập test là 98 hoặc 97

    # X_train sẽ là X[train_index]
    # x_test sẽ là X[test_index]
    # y_train sẽ là y[train_index]
    # y_test sẽ là y[test_index]

    clf_gini.fit(X.iloc[train_index],y.iloc[train_index])
    y_pred = clf_gini.predict(X.iloc[test_index])

    from sklearn.metrics import confusion_matrix
    m1 = confusion_matrix(y[test_index],y_pred, labels=[3,4,5,6,7,8,9])
    #E. Đánh giá mô hình
    print("Đánh giá mô hình ở lần lặp",Fold_index+1,": \n",m1)

    print("Độ chính xác của lớp 3: ",0 if m1[0,0]==0 else m1[0,0]/(m1[0,0]+m1[0,1]+m1[0,2]+m1[0,3]+m1[0,4]+m1[0,5]))
    print("Độ chính xác của lớp 4: ",0 if m1[1,1]==0 else m1[1,1]/(m1[1,0]+m1[1,1]+m1[1,2]+m1[1,3]+m1[1,4]+m1[1,5]))
    print("Độ chính xác của lớp 5: ",0 if m1[2,2]==0 else m1[2,2]/(m1[2,0]+m1[2,1]+m1[2,2]+m1[2,3]+m1[2,4]+m1[2,5]))
    print("Độ chính xác của lớp 6: ",0 if m1[3,3]==0 else m1[3,3]/(m1[3,0]+m1[3,1]+m1[3,2]+m1[3,3]+m1[3,4]+m1[3,5]))
    print("Độ chính xác của lớp 7: ",0 if m1[4,4]==0 else m1[4,4]/(m1[4,0]+m1[4,1]+m1[4,2]+m1[4,3]+m1[4,4]+m1[4,5]))
    print("Độ chính xác của lớp 8: ",0 if m1[5,5]==0 else m1[5,5]/(m1[5,0]+m1[5,1]+m1[5,2]+m1[5,3]+m1[5,4]+m1[5,5]))
    print("Độ chính xác của lớp 9: ",0 if m1[6,6]==0 else m1[6,6]/(m1[6,0]+m1[6,1]+m1[6,2]+m1[6,3]+m1[6,4]+m1[6,5]))
    #E. Độ chính xác tổng thể mỗi lần lặp
    print("Độ chính xác tổng thể của lần lặp thứ",Fold_index+1,":", accuracy_score(y.iloc[test_index],y_pred)*100)
    Sum_AS = Sum_AS + accuracy_score(y.iloc[test_index],y_pred)*100
    Fold_index = Fold_index+1
    print("=================================================================")
#E. Độ chính xác tổng thể trung bình
print("Độ chính xác tổng thể trung bình là: ", Sum_AS/Fold_n)
#Kết quả : 50,3864%

#Xây dựng giải thuật KNN và bayes

#Thực hiện nghi thức
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=1/3,random_state=0)

#Mô hình KNN
from sklearn.neighbors import KNeighborsClassifier
Mohinh_KNN = KNeighborsClassifier(n_neighbors=60)
Mohinh_KNN.fit(x_train,y_train)
y_predKNN = Mohinh_KNN.predict(x_test)
print("Độ chính xác tổng thể của mô hình KNN: ", accuracy_score(y_test,y_predKNN)*100)
#Kết quả độ chính xác tổng thể của mô hình KNN(K=60): 44,335%

#Mô hình Gaussian (Bayes)
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
model = GaussianNB()
model.fit(x_train,y_train)

y_predBayes = model.predict(x_test)
print("Độ chính xác tổng thể của mô hình Bayes: ", accuracy_score(y_test,y_predBayes)*100)
#Kết quảđộ chính xác tổng thể của mô hình Gaussian(Bayes): 41,151%
#F. So sánh các kết quả: kết luận mô hình DecisionTree có độ chính xác cao hơn KNN và Bayes

#==========================================================================================
#Bài 2
print("=========================================================================")
print("=============================   Bài 2   =================================")
print("=========================================================================")

#Đọc dữ liệu
dulieu2 = pd.read_csv("Data_bai2.csv",index_col=0,delimiter=",")
print(dulieu2)

#A.Xây dựng mô hình DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
#Lấy 5 phần tử để train
regressor.fit(dulieu2.iloc[:,0:3].values,dulieu2.iloc[:,3].values)
#B.Dự đoán
y_pred=regressor.predict([[135,39,1]])
print("Kết quả dự đoán [[135,39,1]]: ",y_pred)
#Dự báo phần tử mới có thông tin [[135,39,1]] sẽ có nhãn là 1.
