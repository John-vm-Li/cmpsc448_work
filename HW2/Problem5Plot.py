import random
from types import MappingProxyType
import numpy as np
import matplotlib.pyplot as plt
import Problem5
if __name__ == '__main__':
    # Put the code for the plots here, you can use different functions for each
    # part


    d = np.load("data.npy")
    intercept = np.ones((100, 1))
    data = np.concatenate([d, intercept], axis=1)
    x = d[:,0]
    y = d[: ,1]

    f = np.delete(data, 1,1)
    w = np.random.random(2)

    eta = [0.05, 0.1, 0.1, 0.1]
    delta = [0.1, 0.01, 0, 0]
    lam = [0.001, 0.001, 0.001, 0]
    num_iter = [50, 50, 100, 100]

    '''
    n = []
    for i in range(1, 51):
        n.append(i)

    new_w, history_fw = Problem5.bgd_l2(f, y, w, eta[0], delta[0], lam[0], num_iter[0])
    plt.plot(n, history_fw)
    plt.title("Plot1- eta = 0.05, delta = 0.1, lamda = 0.001, num_iter = 50")
    plt.xlabel("Number of Iterations")
    plt.ylabel("History (history_fw)")
    plt.show()

    new_w, history_fw = Problem5.bgd_l2(f, y, w, eta[1], delta[1], lam[1], num_iter[1])
    plt.plot(n, history_fw)
    plt.title("Plot2- eta = 0.1, delta = 0.01, lamda = 0.001, num_iter = 50")
    plt.xlabel("Number of Iterations")
    plt.ylabel("History (history_fw)")
    plt.show()

    n = []
    for i in range(1, 101):
        n.append(i)
    
    new_w, history_fw = Problem5.bgd_l2(f, y, w, eta[2], delta[2], lam[2], num_iter[2])
    plt.plot(n, history_fw)
    plt.title("Plot3- eta = 0.1, delta = 0, lamda = 0.001, num_iter = 100")
    plt.xlabel("Number of Iterations")
    plt.ylabel("History (history_fw)")
    plt.show()

    new_w, history_fw = Problem5.bgd_l2(f, y, w, eta[3], delta[3], lam[3], num_iter[3])
    plt.plot(n, history_fw)
    plt.title("Plot4- eta = 0.1, delta = 0, lamda = 0, num_iter = 100")
    plt.xlabel("Number of Iterations")
    plt.ylabel("History (history_fw)")
    plt.show()
    
#########################################################
    '''
    eta = [1, 1, 1, 1]
    delta = [0.1, 0.01, 0, 0]
    lam = [0.5, 0.1, 0, 0]
    num_iter = [800, 800, 40, 800]

    n = []
    for i in range(1, 801):
        n.append(i)
    
    new_w, history_fw = Problem5.sgd_l2(f, y, w, eta[0], delta[0], lam[0], num_iter[0])
    plt.plot(n, history_fw)
    plt.title("Plot1- eta = 1, delta = 0.1, lamda = 0.5, num_iter = 800")
    plt.xlabel("Number of Iterations")
    plt.ylabel("History (history_fw)")
    plt.show()

    new_w, history_fw = Problem5.sgd_l2(f, y, w, eta[1], delta[1], lam[1], num_iter[1])
    plt.plot(n, history_fw)
    plt.title("Plot2- eta = 1, delta = 0.01, lamda = 0.1, num_iter = 800")
    plt.xlabel("Number of Iterations")
    plt.ylabel("History (history_fw)")
    plt.show()

    new_w, history_fw = Problem5.sgd_l2(f, y, w, eta[3], delta[3], lam[3], num_iter[3])
    plt.plot(n, history_fw)
    plt.title("Plot4- eta = 1, delta = 0, lamda = 0, num_iter = 800")
    plt.xlabel("Number of Iterations")
    plt.ylabel("History (history_fw)")
    plt.show()

    n = []
    for i in range(1, 41):
        n.append(i)
    
    new_w, history_fw = Problem5.sgd_l2(f, y, w, eta[2], delta[1], lam[2], num_iter[2])
    plt.plot(n, history_fw)
    plt.title("Plot3- eta = 1, delta = 0, lamda = 0, num_iter = 40")
    plt.xlabel("Number of Iterations")
    plt.ylabel("History (history_fw)")
    plt.show()


    



    

