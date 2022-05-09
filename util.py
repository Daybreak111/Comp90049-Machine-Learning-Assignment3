import pandas as pd

rawDataPath_train = "tweets-data/train.pkl"
rawDataPath_test = "tweets-data/test.pkl"
rawDataPath_dev = "tweets-data/dev.pkl"
rawDataPath_unlabeled = "tweets-data/unlabeled.pkl"

tfidfDataPath_train = "tfidf/train_tfidf.pkl"
tfidfDataPath_test = "tfidf/test_tfidf.pkl"
tfidfDataPath_dev = "tfidf/dev_tfidf.pkl"
tfidfDataPath_unlabeled = "tfidf/unlabeled_tfidf.pkl"

embedDataPath_train = "sentence-transformers/train_emb.pkl"
embedDataPath_test = "sentence-transformers/test_emb.pkl"
embedDataPath_dev = "sentence-transformers/dev_emb.pkl"
embedDataPath_unlabeled = "sentence-transformers/unlabeled_emb.pkl"


class Util:

    def __init__(self):
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.df_dev = pd.DataFrame()
        self.df_unlabeled = pd.DataFrame()

    def readRawTrainData(self):
        self.df_train = pd.read_pickle(rawDataPath_train)

    def readRawTestData(self):
        self.df_test = pd.read_pickle(rawDataPath_test)

    def readRawDevData(self):
        self.df_dev = pd.read_pickle(rawDataPath_dev)

    def readRawUnlabeledData(self):
        self.df_unlabeled = pd.read_pickle(rawDataPath_unlabeled)

    def readtfidfTrainData(self):
        self.df_train = pd.read_pickle(tfidfDataPath_train)

    def readtfidfTestData(self):
        self.df_test = pd.read_pickle(tfidfDataPath_test)

    def readtfidfDevData(self):
        self.df_dev = pd.read_pickle(tfidfDataPath_dev)

    def readtfidfUnlabeledData(self):
        self.df_unlabeled = pd.read_pickle(tfidfDataPath_unlabeled)

    def readEmbedTrainData(self):
        self.df_train = pd.read_pickle(embedDataPath_train)

    def readEmbedTestData(self):
        self.df_test = pd.read_pickle(embedDataPath_test)

    def readEmbedDevData(self):
        self.df_dev = pd.read_pickle(embedDataPath_dev)

    def readEmbedUnlabeledData(self):
        self.df_unlabeled = pd.read_pickle(embedDataPath_unlabeled)
