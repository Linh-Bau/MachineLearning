import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def Grad(x):
    return 2*x+5*np.cos(x)

def Cost(x):
    return x**2+5*np.sin(x)

def myGD1(eta, x0):
    x=[x0]
    for it in range(100):
        x_new=x[-1] -eta*Grad(x[-1])
        if(abs(Grad(x_new)))<1e-3:
            break
        x.append(x_new)
    return (x,it)


if __name__=='__main__':
    (x1, it1)=myGD1(.5,-5)
    (x2,it2)=myGD1(.5,5)
    print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], Cost(x1[-1]), it1))
    print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], Cost(x2[-1]), it2))

    # Tạo hai đối tượng Figure và Axes cho hai đồ thị
    fig, (ax1, ax2) = plt.subplots(1,2)

    # Thiết lập trục cho đồ thị 1
    ax1.set_xlim([-6, 6])
    ax1.set_ylim([-10, 60])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('GD with initial x1 = -5')

    # Thiết lập trục cho đồ thị 2
    ax2.set_xlim([-6, 6])
    ax2.set_ylim([-10, 60])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('GD with initial x2 = 5')

    #vẽ đồ thị hàm fx
    x=np.arange(-6,6,.05)
    y=Cost(x)
    ax1.plot(x,y,color='blue')
    ax2.plot(x,y,color='blue')

    #tạo animate cho đồ thị 
    draw_2_points_ax1, =ax1.plot([],[],'r-o')
    draw_2_points_ax2, =ax2.plot([],[],'r-o')

    def update(frame):
        if frame < len(x1):
            draw_2_points_ax1.set_data(x1[frame:frame+2],Cost(np.array(x1[frame:frame+2])))
        if frame < len(x2):
            draw_2_points_ax2.set_data(x2[frame:frame+2],Cost(np.array(x2[frame:frame+2])))

        return draw_2_points_ax1,draw_2_points_ax2

    ani=FuncAnimation(fig,update,frames=100,interval=500,blit=True)
    
    #Hiển thị animation
    plt.show()
