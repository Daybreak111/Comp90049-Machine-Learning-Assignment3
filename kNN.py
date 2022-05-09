import numpy as np
import pandas as pd
import evaluation
from sklearn.neighbors import KNeighborsClassifier


def manhattan_distance(fw1, fw2):
    # insert code here
    if len(fw1) != len(fw2):
        return -1

    distance = 0
    for i in range(len(fw1)):
        distance += abs(float(fw1[i]) - float(fw2[i]))
    return distance


class KNN(object):
    def __init__(self, trainData, testData, k):
        self.k = k
        self.neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
        self.trainData = trainData
        self.testData = testData
        self.predict_labels = list()

    def run(self):
        self.train()
        self.predict()
        self.evaluate()

    def train(self):
        self.neigh.fit(self.trainData['TFIDF'].tolist(), self.trainData['Sentiment'].tolist())

    def predict(self):
        self.predict_labels.append(self.neigh.predict(self.testData['TFIDF'].tolist()))
        # print(self.predict_labels[0])


    # def predict(self):
    #     w = 0.00001
    #     for testID, trainItem in self.trainData.iterrows():
    #         distance_list = list()
    #         # print(row['TFIDF'])
    #         for trainID, testItem in self.testData.iterrows():
    #             distance_list.append(manhattan_distance(trainItem['TFIDF'], testItem['TFIDF']))
    #             distance_array = np.array(distance_list)
    #             neighbor_index = np.argsort(distance_array, kind='stable')[:self.k]
    #         train_module = dict()
    #         for i in range(self.k):
    #             key = self.trainData.iloc[neighbor_index[i], 1]
    #             print(key)
    #             if key in train_module:
    #                 train_module[key] += float(1 / (distance_list[neighbor_index[i]] + w))
    #             else:
    #                 train_module[key] = float(1 / (distance_list[neighbor_index[i]] + w))
    #         train_module_sorted = sorted(train_module.items(), key=lambda kv: kv[1], reverse=True)
    #         self.predict_labels.append(train_module_sorted[0][0])
    #         print(testID, " is testing, result is ", train_module_sorted[0][0])

    def evaluate(self):
        test_labels = self.testData["Sentiment"]
        evaluation.evaluate(test_labels, self.predict_labels[0])

