from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import warnings


def evaluate(test_labels, predict_labels):
    warnings.filterwarnings("ignore")
    print("Accuracy is: ", round(accuracy_score(test_labels, predict_labels), 3))
    result = precision_recall_fscore_support(test_labels, predict_labels, average='weighted')
    print("precision: ", result[0])
    print("recall: ", result[1])
    print("fscore: ", result[2], "\n")