# Intro to Computer Vision             Common Image Processing Tools
# Dvir Ben Asuli                       318208816
# The Open University                  2022-2023
import math

import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
FONTS = ["ALEX", "OPENSANS", "SENSATIONS", "UBUNTOMONO", "TITILIUM"]


# Splitting a list by ratio
def split_set(list, ratio):
    length = len(list)
    mid = int(length * ratio)
    return [list[:mid], list[mid:]]


# Plotting test loss and train loss graphs
def plot_graphs(train_loss, test_loss, train_accu, test_accu):
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.title('Loss')
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(train_accu, label="Train Accuracy")
    plt.plot(test_accu, label="Test Accuracy")
    plt.title('Accuracy')
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.show()


# Generating multi-class roc curve for test set
def generate_roc(labels, predictions):
    num_of_classes = 5
    test_y = []
    y_prediction = []
    tpr, fpr, roc_auc = ([[]] * num_of_classes for _ in range(3))
    f, ax = plt.subplots()

    for i in range(num_of_classes):
        for j in range(len(labels)):
            # Adjusting Softmax output format to ROC Format
            test_y_single = np.array(labels[j], dtype=object).astype(float)
            y_prediction_single = np.array(predictions[j] / max(np.array(predictions[j])), dtype=object).astype(float)

            test_y.append(test_y_single)
            y_prediction.append(y_prediction_single)

        fpr[i], tpr[i], _ = roc_curve(np.array(test_y)[:, i], np.array(y_prediction)[:, i])

        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], label="ROC Curves")

    fonts = ["Alex", "Open-Sans", "Sensations", "Ubunto-Mono", "Titilium"]
    plt.legend(['' + d + ', (AUC = %0.2f)' % roc_auc[fonts.index(d)] for d in fonts])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


# Printing all the image names and the fonts that were currectly identified in them
def print_results(results_dict):
    print("Results:")

    for key in results_dict.keys():
        fonts_str = ""

        for item in results_dict[key]:
            fonts_str = fonts_str + str(item) + " "
        print(str(key) + ": " + fonts_str)

# Saving the trained model to pickle file
def save_model(model):
    pickle.dump(model, open('model.pkl', 'wb'))

# Loading a model from pickle file
def load_model(filename):
    pickled_model = pickle.load(open(filename, 'rb'))
    return pickled_model

