# vi du 3
print("-------------------------------------------------------------------------")
print("Vi du 3")
print("-------------------------------------------------------------------------")
a=5
b=3
if a>b:
    a=a*2+3
    b=b-6
    c= a/6
    print("Ket qua phep tinh 1: ",c)
c = a+b+\
    10*a-b/4-\
    5+a*3
print("Ket qua phep tinh 2: ",c)

# vi du 4
print("-------------------------------------------------------------------------")
print("Vi du 4")
print("-------------------------------------------------------------------------")
#lenh if
a=5
b=3
print("Ket qua sau khi thuc thi lenh if a>b")
if a>b :
    print("True")
    print(a)
else:
    print("False")
    print(b)

#lenh while
a=1
b=10
print("In gia tri voi while")
while a<b:
    a+=1
    print(a)

#lenh for
print("In gia tri for")
for i in range (1,10):
    print(i)

#khai bao ham
def binhphuong(number):
    return number*number
print("Binh phuong cua 5: ",binhphuong(5))

# vi du 5
# Khai bao kieu du lieu
print("-------------------------------------------------------------------------")
print("Vi du 5")
print("-------------------------------------------------------------------------")
a = 5
b = -7
c = 1.234

str1 = "Hello"
str2 = "welcome"
str3 = "abcdef12345"

cats = ['Tom', 'Snappy', 'Kitty', 'Jessie', 'Chester']

print("In phan tu thu 2: ",cats[2])
print("In phan tu cuoi: ",cats[-1])
print("In phan tu tu 1-3: ",cats[1:3])
print("In tat ca cac phan tu: ",cats)


cats.append("Jerry")
print("Mang sau khi duoc them Jerry: ",cats)

cats[-1] = "Jerry Cat"
print("Doi ten Jerry thanh Jerry Cat: ",cats)

del cats[1]
print("Mang sau khi xoa phan tu thu 1: ",cats)

dict = {"Name" : "Zyra", "Age" : 7, "Class" : "A5"}
print ("Gia tri cua key Name: ",dict["Name"])
print ("Gia tri cua key Age: ",dict["Age"])

dict["Age"] = 8
print ("Sua gia tri cua key Age 7->8: ",dict["Age"])

# vi du 6
# cai dat thu vien
print("-------------------------------------------------------------------------")
print("Vi du 6")
print("-------------------------------------------------------------------------")
import  numpy as np

a = np.array([0,1,2,3,4,5])
print("In mang a: ",a)
print("So chieu cua mang a: ",a.ndim)
print("Hien thi hinh dang cua a: ",a.shape)
a[a>3] =10
print("Gan cac phan tu co chi so > 3 bang 10:",a)

b= a.reshape(3,2)
print("Doi dang cua a từ (6,) sang (3,2)\n",b)
print("In ra b[1][1]: ",b[1][1])
b[2][0]=50

c = b*2
print("Mang sau khi dc nhan 2: \n",c)

import pandas as pd
dt = pd.read_csv('play_tennis2.csv', delimiter=";")
print("File sau khi duoc doc \n",dt)
print("\nNam dong dau tien")
print(dt.head())
print("\n7 dong cuoi cung")
print(dt.tail(7))
print("\nHang thu 3 den hang thu 5")
print(dt.loc[3:5])
print("\nCot 3 den cot 5")
print(dt.iloc[:,3:6])
print("\nHang 5 den hang 9 cua cot 3")
print(dt.iloc[5:10,3:4])
print("\nIn ra cac gia tri o cot co ten Windy ")
print(dt.Windy.unique())
