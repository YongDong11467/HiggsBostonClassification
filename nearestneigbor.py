import numpy as np
import numpy.linalg as la

def run(X,y,z):
    (n,d) = np.shape(X)
    c = 0
    # print(np.array([X[0]]).T)
    b = la.norm(z - np.array([X[0]]).T)
    for t in range(1, n):
        if la.norm(z - np.array([X[t]]).T) < b:
            c = t
            b = la.norm(z - np.array([X[t]]).T)

    label = y[c][0]
    return label