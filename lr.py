from sklearn.linear_model import LogisticRegression
import evaluation


class LR(object):
    def __init__(self, trainData, testData):
        self.clf = LogisticRegression()
        self.trainData = trainData
        self.testData = testData
        self.predict_labels = list()

    def run(self):
        self.train()
        self.predict()
        self.evaluate()

    def train(self):
        self.clf.fit(self.trainData['TFIDF'].tolist(), self.trainData['Sentiment'].tolist())

    def predict(self):
        self.predict_labels.append(self.clf.predict(self.testData['TFIDF'].tolist()))
        # print(self.predict_labels[0])

    def evaluate(self):
        test_labels = self.testData["Sentiment"]
        evaluation.evaluate(test_labels, self.predict_labels[0])
