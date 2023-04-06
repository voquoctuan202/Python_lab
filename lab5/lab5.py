# #Perceptron
#
# import numpy as np
# import matplotlib.pyplot as plt
# #
# # X = np.array([[0,0,1,1],[0,1,0,1]])
# # X
# # X=X.T
# #
# # X1 = np.array([[0,0],[0,1],[1,0],[1,1]])
# # X1
# #
# # Y = np.array([0,0,0,1])
# # Y
# #
# # colormap = np.array(['red','green'])
# # plt.axis([0,1.5,0,2])
# # plt.scatter(X[:,0],X[:,1],c=colormap[Y],s=150)
# # plt.xlabel("Gia tri thuoc tinh X1")
# # plt.ylabel("Gia tri thuoc tinh X2")
# # plt.show()
# #
# # def my_perceptron(X,y,eta,lanlap):
# #     n = len(X[0,])
# #     m = len(X[:,0])
# #     print("m =",m,"n =",n)
# #     w0 = -0.2
# #     w = (0.5,0.5)
# #     print ("w0 = ",w0)
# #     print ("w= ",w)
# #     for t in range(0,lanlap):
# #         print("lanlap ___",t+1)
# #         for i in range(0,m):
# #             gx = w0 + sum(X[i,]*w)
# #             print ("gx = ",gx)
# #             if(gx>0):
# #                 output = 1
# #             else:
# #                 output = 0
# #             w0 = w0 + eta*(y[i]-output)
# #             w = w + eta*(y[i]-output)*X[i,]
# #             print ("w0 =",w0)
# #             print (" w =",w)
# #     return (np.round(w0,3),np.round(w,3))
# #
# # print (my_perceptron(X,Y,0.15,2))
#
# #Bai tap 2
# import pandas as pd
# dt = pd.read_csv("data_per.csv",delimiter=",")
# #print(dt)
#
# #Nghi thức Hold_out
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(dt.iloc[:,0:5],dt.iloc[:,5], test_size=1/3.0,)
#
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train.iloc[:,0:2] = sc.fit_transform(X_train.iloc[:,0:2])
# X_test.iloc[:,0:2] = sc.fit_transform(X_test.iloc[:,0:2])
# print(X_train)
# from sklearn.linear_model import Perceptron
# net = Perceptron()
#
# net.fit(X_train,y_train)
# print("Net.conef_", net.coef_)
# print("Net.intercept_", net.intercept_)
# print("Net.n_iter_", net.n_iter_)
# y_pred =  net.predict(X_test)
#
# from sklearn.metrics import accuracy_score
# print("Độ chính xác: ",accuracy_score(y_test,y_pred))


#Clustering

# import matplotlib.pyplot as plt
#
# x = [4,5,10,4,3,11,14,6,10,12]
# y = [21,19,24,17,16,25,24,22,21,21]
#
# plt.scatter(x,y)
# plt.show()
#
# from sklearn.cluster import KMeans
#
# data = list(zip(x,y))
# inertias = []
#
# for i in range(1,11):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(data)
#     inertias.append(kmeans.inertia_)
#
# plt.plot(range(1,11), inertias , marker='o')
# plt.title("Elbow method")
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()
#
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(data)
#
# plt.scatter(x,y,c=kmeans.labels_)
# plt.show()
#
# import numpy as np
#
# from scipy.cluster.hierarchy import dendrogram, linkage
#
# data = list(zip(x,y))
# linkage_data = linkage(data, method='ward', metric= 'euclidean')
# dendrogram(linkage_data)
#
# plt.show()

#Bai tap Clustering
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("USArrests.csv",delimiter=",",index_col=0)
print (df)
from sklearn.preprocessing import scale
df = pd.DataFrame(scale(df), index=df.index, columns=df.columns)

from sklearn.cluster import KMeans

inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias , marker='o')
plt.title("Elbow method")
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

from sklearn.decomposition import PCA

# Khởi tạo đối tượng PCA với số comp = 2
my_pca = PCA (n_components = 2 )

# Fit vào data
my_pca.fit(df)

# Thực hiện transform
#df_2_dim = my_pca.transform(df)

df_2_dim = pd.DataFrame(my_pca.transform(df), index=df.index)
print(df_2_dim.head())
kmeans = KMeans(n_clusters=4)
kmeans.fit(df)

plt.scatter(df_2_dim.iloc[:,0],df_2_dim.iloc[:,1],c=kmeans.labels_)
plt.show()
