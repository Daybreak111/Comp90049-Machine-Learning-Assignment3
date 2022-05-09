import numpy as np
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
import evaluation


class SemiSupervise(object):
    def __init__(self, rawData, labeledData, testData):
        svc = SVC(probability=True, gamma="auto")
        self.clf = SelfTrainingClassifier(svc)
        self.rawData = rawData.sample(n=5000)
        self.labeledData = labeledData.sample(n=1000)
        self.testData = testData.sample(n=10)
        self.predict_labels = list()

    def run(self):
        self.train()
        self.predict()
        self.evaluate()

    def train(self):
        trainData = list()
        trainLabel = list()

        for idx, row in self.labeledData.iterrows():
            trainData.append(row['TFIDF'])
            if row['Sentiment'] == 'positive':
                trainLabel.append(1)
            elif row['Sentiment'] == 'negative':
                trainLabel.append(0)
        for idx, row in self.rawData.iterrows():
            trainData.append(row['TFIDF'])
            trainLabel.append(-1)

        self.clf.fit(trainData, trainLabel)

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

