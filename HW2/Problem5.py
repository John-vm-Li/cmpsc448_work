import math
import random
import numpy as np
from numpy.lib.function_base import _gradient_dispatcher

# function to calculate value of the derivative of g function
def gradient_derivative(y,x,w, delta):
    g = 0
    # based on function g calculate the derivative of g
    for i in range(len(x)):
        if(y[i] >= (np.dot(w,x[i]) + delta)):
            g += (-2) * (y[i] - np.dot(w,x[i]) - delta)*x[i]
            
        elif(abs(y[i] - np.dot(w,x[i])) < delta):
            g += 0
            
        elif(y[i] <=  (np.dot(w,x[i]) - delta)):
            g += (-2) * (y[i] - np.dot(w,x[i]) + delta) *x[i]
    
    # divide the summation of the derivatives of each g value by n
    g_d = g/len(x)
    return g_d

# function to calculate value of g function
def gradient(y,x,w, delta):
    g = 0
    # based on function g calculate the value of g
    for i in range(len(x)):
        if(y[i] >= (np.dot(w,x[i]) + delta)):
            g += math.pow((y[i] - np.dot(w,x[i]) - delta), 2)
            
        elif(abs(y[i] - np.dot(w,x[i])) < delta):
            g += 0
            
        elif(y[i] <=  (np.dot(w,x[i]) - delta)):
            g += math.pow((y[i] - np.dot(w,x[i]) + delta), 2) 

    # divide the summation  each g value by n
    g_fn = g/len(x)
    return g_fn

# calculate the derivative of f
def f_derivative(y,data,w_t, delta,lam):
    # get the final value of derivative of g function 
    g_f_d = gradient_derivative(y,data, w_t, delta)
    # get derivative of lambda term
    lam_f_d = 2 * lam* np.sum(w_t)
    # calculate f prime val 
    f_derivative = g_f_d + lam_f_d

    return f_derivative

# calculate the f val
def fn(y,data,w_t, delta,lam):
     # get the final value of g function 
    g_f = gradient(y,data, w_t, delta)
    # get value of lambda term
    lam_f = lam*np.sum(w_t**2)
    # calculate f val 
    f = g_f + lam_f

    return f

# function to  optimize the a objective function using gradient descent
def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    history_fw = []

    new_w = w
    w_t = np.transpose(new_w)
    g = 0
    g_d = 0
    # calculate f value and update weights over number of iterations 
    for i in range(num_iter):
        # calculate derivative of f
        f_d = f_derivative(y,data,w_t, delta,lam)
        # update weights 
        new_w = new_w - (eta*f_d)
        w_t = np.transpose(new_w) 
        # calculate f val
        f = fn(y,data,w_t, delta,lam)
        # store history of f value 
        history_fw.append(f)


    return new_w, history_fw


# function to calculate value of the derivative of g function for SGD
def SGD_gradient_derivative(y,x,w, delta):
    g = 0
    # based on function g calculate the derivative of g
    # for SGD we need to sample a small subset of training data uniformly at random
    subset = 70
    for i in range(subset):
        r = random.randrange(0,100)
        if(y[r] >= (np.dot(w,x[r]) + delta)):
            g += (-2) * (y[r] - np.dot(w,x[r]) - delta)*x[r]
            
        elif(abs(y[r] - np.dot(w,x[r])) < delta):
            g += 0
            
        elif(y[r] <=  (np.dot(w,x[r]) - delta)):
            g += (-2) * (y[r] - np.dot(w,x[r]) + delta) *x[r]

    # divide the summation of the derivatives of each g value by n
    g_d = g/len(x)
    return g_d

# function to calculate value of g function for SGD
def SGD_gradient(y,x,w, delta):
    g = 0
    
    # based on function g calculate the value of g
    # for SGD we need to sample a small subset of training data uniformly at random
    subset = 70
    for i in range(subset):
        r = random.randrange(0,100)
        if(y[r] >= (np.dot(w,x[r]) + delta)):
            g += math.pow((y[r] - np.dot(w,x[r]) - delta), 2)
            
        elif(abs(y[r] - np.dot(w,x[r])) < delta):
            g += 0
            
        elif(y[r] <=  (np.dot(w,x[r]) - delta)):
            g += math.pow((y[r] - np.dot(w,x[r]) + delta), 2) 

    # divide the summation  each g value by n
    g_fn = g/len(x)
    return g_fn

def SGD_f_derivative(y,data,w_t, delta,lam):
    # get the final value of derivative of g function 
    g_f_d = SGD_gradient_derivative(y,data, w_t, delta)
    # get derivative of lambda term
    lam_f_d = 2 * lam* np.sum(w_t)
    # calculate f prime val 
    f_derivative = g_f_d + lam_f_d

    return f_derivative

# calculate the f val
def SGD_fn(y,data,w_t, delta,lam):
    # get the final value of g function 
    g_f = SGD_gradient(y,data, w_t, delta)
    # get value of lambda term
    lam_f = lam*np.sum(w_t**2)
    # calculate f val
    f = g_f + lam_f

    return f

# function to  optimize the a objective function using stochatic gradient descent
def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
    history_fw = []

    new_w = w
    w_t = np.transpose(new_w)
    g = 0
    g_d = 0
    if(i != -1): 
        num_iter = 1
    # calculate f value and update weights over number of iterations 
    for i in range(num_iter):  
        # calculate derivative of f      
        f_d = SGD_f_derivative(y,data,w_t, delta,lam)
        # use new laerning rate of eta/ sqrt(i)
        learning = eta/((i+1)**0.5)
        # update weights
        new_w = new_w - (learning*f_d)
        w_t = np.transpose(new_w) 
        # calculate f val
        f = SGD_fn(y,data,w_t, delta,lam)
        # store history of f value 
        history_fw.append(f)


    
    return new_w, history_fw
