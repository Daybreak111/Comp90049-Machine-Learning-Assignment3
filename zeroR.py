import pandas as pd
import evaluation


class ZeroR(object):
    def __init__(self, trainData, testData):
        self.trainData = trainData
        self.testData = testData
        self.predict_labels = list()
        self.labels = dict()

    def run(self):
        self.train()
        self.predict()
        self.evaluate()

    def train(self):
        for item in self.trainData["Sentiment"]:
            if item not in self.labels:
                self.labels[item] = 1
            else:
                self.labels[item] += 1
        # print(self.labels)

    def predict(self):
        max_count = 0
        label = ''

        for key, value in self.labels.items():
            if value > max_count:
                max_count = value
                label = key

        for item in self.testData["Sentiment"]:
            self.predict_labels.append(label)

    def evaluate(self):
        test_labels = self.testData["Sentiment"]
        evaluation.evaluate(test_labels, self.predict_labels)
