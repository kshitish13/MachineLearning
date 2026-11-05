import numpy as np

def gradient_descent(x, y, lr=0.01, epochs=5000):
    m, b = 0.0, 0.0

    for  epoch in range(epochs+1):
        y_pred=m*x+b
        error= y_pred - y
        cost=np.mean(error**2)

        dm= 2*np.mean(error*x) #partial derivative wrt m
        db=2*np.mean(error) #partial derivative wrt b

        b=b-lr*db #updating b
        m=m-lr*dm #updating m

        if epoch%100==0:
            print(f"m = {m}, b = {b}, Epoch={epoch} : Cost = {cost}")


if __name__== "__main__":
    x=np.array([1,2,3,4,5])
    y=np.array([5,7,9,11,13])

    gradient_descent(x,y)