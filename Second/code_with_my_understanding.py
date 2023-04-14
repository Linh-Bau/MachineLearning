import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from matplotlib.widgets import Button


#tạo ra bộ dữ liệu
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, 20)
X1 = np.random.multivariate_normal(means[1], cov, 100)
X2 = np.random.multivariate_normal(means[2], cov, 1000)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3


# thuật toán
# cho k điểm cemter bất kỳ
# từ k điểm tâm đó tìm ra được center là điểm gần nó nhất, ở đây có nghĩa là ta đã tìm được Y
# từ Y, ta tính lại M, nếu M không thay đổi nghĩa là ta đã tìm được tọa độ center chính xác

def kmeans_get_ramdom_center(X,K):
    choose_col= np.random.choice(X.shape[0],K,replace=False)
    return X[choose_col,:]

#M đã được cố định cần tìm Y
#với mỗi điểm xi thuộc X, ta cần tìm j là chỉ số của centers, sao cho xi gần centers nhất
def kmeans_assign_labels(X,centers):
    D=cdist(X,centers)
    #hàm này trả về 1 ma trận là khoảng cách tới mỗi điểm center
    return np.argmin(D,axis=1)


#tính center dựa trên labels
def kmeans_update_centers(X,labels,K):
    centers=np.zeros((K,X.shape[1]))
    for k in range(K):
        #tính tổng các point có label bằng K
        k_cluster_points=X[labels==k,:]
        centers[k,:]=np.mean(k_cluster_points,axis=0)
    return centers

#so sánh 2 centers cũ và mới xem có bằng nhau không
def kmeans_converged(old_centers, new_centers):
    if ((set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))):
        return True
    return False


color=['ro','go','bo']

#in ra màn hình kết quả
def mpl_show_cluser(X,Labels,Centers,K):
    subplot_col=1
    subplot_row=Labels.shape[0]
    for i in range(1,subplot_row):
        plt.subplot(subplot_row,subplot_col,i)
        plt.title('step {}'.format(i))
        #plot center
        for c in Centers[:,:,i]:
            plt.plot(c[0],c[1],'y^',markersize = 4, alpha = .8)
        #plot data
        for k in range(K):
            k_cluser_point=X[Labels[i,:]==k,:]
            plt.plot(k_cluser_point[:,0],k_cluser_point[:,1],color[k],markersize = 4, alpha = .8)
    plt.show()


class mpl_data_updater:

    def __init__(self,X,Labels,Centers,K) -> None:
        self.fig, self.ax=plt.subplots()
        self.fig.subplots_adjust(bottom=0.2)
        self.step=0
        self.max_step=Labels.shape[0]
        self.X=X
        self.Labels=Labels
        self.Centers=Centers
        self.K=K
        axprev = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev)
        self.__update_data()
        plt.show()

    def __update_data(self):
        self.ax.cla()
        self.fig.suptitle("step {}/{}".format(self.step+1,self.max_step))

        for k in range(K):
            k_cluser_point=X[self.Labels[self.step,:]==k,:]
            self.ax.plot(k_cluser_point[:,0],k_cluser_point[:,1],color[k],markersize = 4, alpha = .5)

        for c in Centers[:,:,self.step]:
            self.ax.plot(c[0],c[1],'y^',markersize = 6, alpha = 1)
        plt.draw()

    def next(self, event):
        if self.step==self.max_step-1:
            return
        self.step+=1
        self.__update_data()
        

    def prev(self, event):
        if self.step==-1:
            return
        self.__update_data()
        self.step-=1

if __name__=="__main__":
    Lables=None
    centers=kmeans_get_ramdom_center(X,K)
    Centers=np.array(centers)
    while True:
        labels=kmeans_assign_labels(X,centers)
        new_centers=kmeans_update_centers(X,labels,K)
        if Lables is None:
            Lables=np.array(labels)
        else:
            Lables=np.vstack((Lables,labels))
        if kmeans_converged(centers,new_centers):
            break
        centers=new_centers
        Centers= np.dstack((Centers,centers))
    print("clusters center: ",centers)
    figure=mpl_data_updater(X,Lables,Centers,K)    
