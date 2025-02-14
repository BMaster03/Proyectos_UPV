#Training_data
#from statistics import covariance
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_gaussian_quantiles #type: ignore

#Create datasets from zero - For in example of clasification 

N = 1000

gaussian_quantiles = make_gaussian_quantiles(
    mean = None,
    cov = 0.1,
    n_samples = N,
    n_features = 2, 
    n_classes = 2,
    shuffle = True,
    random_state = None
)

x, y = gaussian_quantiles

print(x.shape)
print(y.shape)

plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.viridis) 
plt.show()

def Sigmoid(X, derivate = False): #me return of derivative of fuction sigmoid
    if derivate:
        return np.exp(-x)/(np.exp(-x)+1) ** 2
    else:
        return 1/(1 + np.exp(-x))
    
def Relu(x, derivative = False):
    if derivative:
        x [x <= x] = 0
        x [x > 0 ] = 1
        return x
    else:
        return np.maximum(0,x)

# Loss of Fuction 

def Mse(y,y_hat, derivative = False):
    if derivative:
        return (y_hat - y)
    else:
        return np.mean((y_hat - y)) ** 2





    