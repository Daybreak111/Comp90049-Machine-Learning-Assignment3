from NB import NaiveBayes
from kNN import KNN
from kmeans import Kmeans
from lr import LR
from semiSupervise import SemiSupervise
from util import Util
from zeroR import ZeroR

if __name__ == '__main__':
    util = Util()
    util.readEmbedTrainData()
    util.readEmbedTestData()
    util.readEmbedDevData()
    util.readEmbedUnlabeledData()

    # print("------------------------")
    # print("Baseline 0-R:")
    # zeroR = ZeroR(util.df_train, util.df_dev)
    # zeroR.run()
    # print("")
    #
    # print("------------------------")
    # k = 13
    # print("KNN - ", k, ":")
    # knn = KNN(util.df_train, util.df_dev, k)
    # knn.run()
    # print("")
    #
    # print("------------------------")
    # print("Naive Bayes Gaussian:")
    # nb = NaiveBayes(util.df_train, util.df_dev)
    # nb.run()
    # print("")
    #
    # print("------------------------")
    # print("Logistic Regression:")
    # lr = LR(util.df_train, util.df_dev)
    # lr.run()
    # print("")
    #
    # print("------------------------")
    # print("Unsupervised Learning kMeans:")
    # km = Kmeans(util.df_unlabeled, util.df_dev)
    # km.run()
    # print("")

    print("------------------------")
    print("Semi-supervised Learning:")
    semi = SemiSupervise(util.df_unlabeled, util.df_train, util.df_dev)
    semi.run()
    print("")

