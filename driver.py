import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import nearestneigbor

rawData = pd.read_csv (r'./higgs-boson/training.csv')
# rawtest = pd.read_csv (r'./higgs-boson/test.csv')

# TP, TN, FP, FN
C_roc = []

# TP, TN, FP, FN
N_roc = []

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

def kfoldcrossvalid(k, X, y, C, nFlag):
    (n, d) = np.shape(X)
    kf_err = np.zeros(k)
    if nFlag:
        alg = KNeighborsClassifier(n_neighbors=C)
    else:
        alg = SVC(C=C, kernel='linear')

    for i in range(0, k):
        T = set(range(int(np.floor((n * i) / k)), int(np.floor(((n * (i + 1)) / k) - 1)) + 1))
        S = set(range(0, n)) - T

        alg.fit(X[list(S)], y[list(S)])
        kf_err[i] = np.mean(y[list(T)] != alg.predict(X[list(T)]))
    CM = confusion_matrix(y, alg.predict(X))

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    if nFlag:
        N_roc.append([TP, TN, FP, FN])
    else:
        C_roc.append([TP, TN, FP, FN])
    err = np.mean(kf_err)
    return err

def main():
    trainData, testData, validationData, trainLabel, testLabel, validationLabel = processData()
    # print(processData())
    # acc = 0
    # for i in range(0, testData.shape[0]):
    #     label = nearestneigbor.run(trainData, trainLabel, testData[i])
    #     if label == testLabel[i][0]:
    #         acc = acc + 1
    # print(acc / 1000)

    C_list = [0.1, 0.2, 0.3, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    C_list_acc = []

    N_list = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    N_list_acc = []

    k = 10
    best_C = 0.0
    best_err = 1.1
    trainLabel = np.ravel(trainLabel.reshape(-1))
    for C in C_list:
        err = kfoldcrossvalid(k, trainData, trainLabel, C, nFlag=False)
        C_list_acc.append(1 - err)
        print("C=", C, ", err=", err)
        if err < best_err:
            best_err = err
            best_C = C
    print("best_C=", best_C)

    best_N = 0.0
    best_err = 1.1
    for N in N_list:
        err = kfoldcrossvalid(k, trainData, trainLabel, N, nFlag=True)
        N_list_acc.append(1 - err)
        print("N=", N, ", err=", err)
        if err < best_err:
            best_err = err
            best_N = N
    print("best_N=", best_N)
    print("Finish Preprocessing")

    plt.plot(C_list, C_list_acc, label='Hyper_Accuracy_Plot(C)')
    plt.xlabel('Hyperparameter C')
    plt.ylabel("accuracy (%)")
    plt.axis([0, 1, .6, 1])
    plt.savefig("Hyper_Accuracy_Plot(C).pdf")
    plt.show()

    plt.plot(N_list, N_list_acc, label='Hyper_Accuracy_Plot(N)')
    plt.xlabel('Hyperparameter N')
    plt.ylabel("accuracy (%)")
    plt.axis([0, 18, .6, 1])
    plt.savefig("Hyper_Accuracy_Plot(N).pdf")
    plt.show()

    print(C_roc)
    print(N_roc)

    x = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, len(x)):
        d = C_roc[i]
        x[i] = d[1]/(d[1] + d[2])
        y[i] = d[0]/(d[0] + d[3])
    plt.plot(x, y, label='ROC_Plot(C)')
    plt.xlabel('Specificity')
    plt.ylabel("Sensitivity")
    plt.axis([0, 1, 0, 1])
    plt.savefig("ROC_Plot(C).pdf")
    plt.show()

    x = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, len(x)):
        d = N_roc[i]
        x[i] = d[1]/(d[1] + d[2])
        y[i] = d[0]/(d[0] + d[3])
    plt.plot(x, y, label='ROC_Plot(N)')
    plt.xlabel('Specificity')
    plt.ylabel("Sensitivity")
    plt.axis([0, 1, 0, 1])
    plt.savefig("ROC_Plot(N).pdf")
    plt.show()


main()