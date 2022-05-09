from sklearn.cluster import KMeans
import evaluation


class Kmeans(object):
    def __init__(self, trainData, testData):
        self.clf = KMeans(n_clusters=2, random_state=0)
        self.trainData = trainData
        self.testData = testData
        self.predict_labels = list()

    def run(self):
        self.train()
        self.predict()
        self.evaluate()

    def train(self):
        self.clf.fit(self.trainData['TFIDF'].tolist())

    def predict(self):
        self.predict_labels.append(self.clf.predict(self.testData['TFIDF'].tolist()))
        # print(self.predict_labels[0])

    def evaluate(self):
        test_labels = self.testData["Sentiment"]
        predictLabels = self.predict_labels[0]
        predicLabels_trans = list()

        for i in range(len(predictLabels)):
            if predictLabels[i] == 1:
                predicLabels_trans.append("negative")
            elif predictLabels[i] == 0:
                predicLabels_trans.append("positive")

        evaluation.evaluate(test_labels, predicLabels_trans)
