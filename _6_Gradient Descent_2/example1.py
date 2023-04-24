import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def has_convergend(theta_new,grad):
    return np.linalg.norm(grad(theta_new))/len(theta_new)<1e-3


def GD_momentum(theta_init,grad,eta,gamma):
    theta=[theta_init]
    v_old=np.zeros_like(theta_init)
    for it in range(1000):
        v_new=gamma*v_old+eta*grad(theta[-1])
        theta_new=theta[-1]-v_new
        if has_convergend(theta_new,grad):
            break
        theta.append(theta_new)
        v_old=v_new
    return (theta,it)


if __name__=='__main__':
    def grad(x):
        return 2*np.array(x)+10*np.cos(np.array(x))

    theta_init=[-5]
    (theta,it)=GD_momentum(theta_init,grad,0.1,0.9)
    print('result: ', theta[-1],',',it)

    #vẽ đồ thị biểu diễn
    fig, ax=plt.subplots(1,1)
    x_data=np.arange(-4,6,0.05)
    y_data=x_data**2+10*np.sin(x_data)
    ax.plot(x_data,y_data,'b-')

    two_point, =ax.plot([],[],'r-o')
    def update(frame):
        if frame<len(theta)-1:
            current_theta=np.array(theta[frame:frame+2])
            two_point.set_data(current_theta,current_theta**2+10*np.sin(current_theta))
        return two_point,
        
    ani= FuncAnimation(fig,update,frames=200,interval=100,blit=True)
    plt.show()

