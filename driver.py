import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import nearestneigbor

rawData = pd.read_csv (r'./higgs-boson/training.csv')
# rawtest = pd.read_csv (r'./higgs-boson/test.csv')

def processData():
    data = rawData[:1000]
    trainData = data[:400]
    testData = data[400:700]
    validationData = data[700:1000]

    trainLabel = trainData.iloc[:,-1:]
    testLabel = testData.iloc[:,-1:]
    validationLabel = validationData.iloc[:,-1:]

    trainData = trainData.iloc[:, :-1]
    testData = testData.iloc[:, :-1]
    validationData = validationData.iloc[:, :-1]
    return trainData.values, testData.values, validationData.values, trainLabel.values, testLabel.values, validationLabel.values

# def kfoldcrossvalid(B,X_subset,y_subset,C):
#     n = len(X_subset)
#     bs_err = np.zeros(B)
#     for b in range(B):
#         train_samples = list(np.random.randint(0,n,n))
#         test_samples = list(set(range(n)) - set(train_samples))
#         alg = SVC(C=C,kernel='linear')
#         alg.fit(X_subset[train_samples], y_subset[train_samples])
#         bs_err[b] = np.mean(y_subset[test_samples] != alg.predict(X_subset[test_samples]))
#     err = np.mean(bs_err)
#     return err


def kfoldcrossvalid(k, X, y, C, nFlag):
    (n, d) = np.shape(X)
    kf_err = np.zeros(k)
    for i in range(0, k):
        T = set(range(int(np.floor((n * i) / k)), int(np.floor(((n * (i + 1)) / k) - 1)) + 1))
        S = set(range(0, n)) - T

        if nFlag:
            alg = KNeighborsClassifier(n_neighbors=C)
        else:
            alg = SVC(C=C, kernel='linear')
        alg.fit(X[list(S)], y[list(S)])
        kf_err[i] = np.mean(y[list(T)] != alg.predict(X[list(T)]))
    err = np.mean(kf_err)
    return err

# def kfoldcrossvalid(k, X, y, C):
#     (n, d) = np.shape(X)
#     kf_err = np.zeros(k)
#     for i in range(0, k):
#         T = set(range(int(np.floor((n * i) / k)), int(np.floor(((n * (i + 1)) / k) - 1)) + 1))
#         S = set(range(0, n)) - T
#
#         alg = KNeighborsClassifier(n_neighbors=C)
#         alg.fit(X[list(S)], y[list(S)])
#         kf_err[i] = np.mean(y[list(T)] != alg.predict(X[list(T)]))
#     err = np.mean(kf_err)
#     return err

def main():
    trainData, testData, validationData, trainLabel, testLabel, validationLabel = processData()
    # print(processData())
    # acc = 0
    # for i in range(0, testData.shape[0]):
    #     label = nearestneigbor.run(trainData, trainLabel, testData[i])
    #     if label == testLabel[i][0]:
    #         acc = acc + 1
    # print(acc / 1000)

    C_list = [0.1, 0.3, 0.5, 0.7]
    N_list = [1, 3, 5, 7]
    k = 10
    best_C = 0.0
    best_err = 1.1
    trainLabel = np.ravel(trainLabel.reshape(-1))
    for C in C_list:
        err = kfoldcrossvalid(k, trainData, trainLabel, C, nFlag=False)
        print("C=", C, ", err=", err)
        if err < best_err:
            best_err = err
            best_C = C
    print("best_C=", best_C)

    best_N = 0.0
    best_err = 1.1
    for N in N_list:
        err = kfoldcrossvalid(k, trainData, trainLabel, N, nFlag=True)
        print("N=", N, ", err=", err)
        if err < best_err:
            best_err = err
            best_N = N
    print("best_N=", best_N)
    print("Finish Preprocessing")


main()