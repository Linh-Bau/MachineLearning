import numpy as np
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T


#tạo ma trận với trọng số là 1
ones=np.ones((X.shape[0],1))
#tạo ma trận Xbar bằng cách ghép 2 cột, cột đầu tiên là trọng số 1, cột thứ 2 là x1
Xbar=np.concatenate((ones,X),axis=1)

#tính toán
b=np.dot(Xbar.T,y)
A=np.dot(Xbar.T,Xbar)

w=np.dot(np.linalg.pinv(A),b)

print("Ma trận tối ưu w thu được: ",w)


#tính giá trị sẽ thu được khi dùng công thức này
x0=np.array([(1,140),(1,185)])
y0=np.dot(x0,w)

#in gia tri du lieu thuc te ra man hinh
plt.plot(X.T,y.T,'ro')
#ve line
plt.plot(x0,y0,'g')


#tinh theo thu vien sklearn

repr=linear_model.LinearRegression(fit_intercept=False)
repr.fit(X,y)

w_=repr.coef_
print("Ma trận tối ưu w_ thu được",w_)
x0_=np.array([(1,140),(1,185)])
y0_=repr.predict(x0_)



#ve line
plt.plot(x0_,y0_,'b')

plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
