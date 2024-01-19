# Intro to Computer Vision             Project
# Dvir Ben Asuli                       318208816
# The Open University                  January 2023


from CNN import *
from preprocessing import *

import warnings
import torch.nn as nn
import torch.optim as optim


# Creating, Training and saving model
def build_model():
    print("Extracting database")
    training_and_validation_dataset = build_complete_dataset(DATABASE_PATH + DATABASE_NAME)
    training_dataset, validation_dataset = split_set(training_and_validation_dataset, 0.8)

    training_input, training_labels = build_model_input(training_dataset)
    validation_input, validation_labels = build_model_input(validation_dataset)

    print("Building new model")
    cnn = CNN()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.000001)

    print("Training the model")
    print("")
    train_loss, test_loss, train_accu, test_accu = train([validation_input, validation_labels],
                                                         [training_input, training_labels], cnn, optimizer, loss,
                                                         num_epochs=120)

    plot_graphs(train_loss, test_loss, train_accu, test_accu)

    print("Saving the model")
    save_model(cnn)


# Loading model from pickle and testing it
def test_model():
    print("Extracting database")
    testing_dataset = build_complete_dataset_with_names(DATABASE_PATH + DATABASE_NAME)
    testing_input, testing_labels, testing_names = build_model_input_with_names(testing_dataset)

    print("Loading Model")
    model = load_model('model.pkl')
    loss = nn.CrossEntropyLoss()

    print("Testing Model")
    print("")
    test_accu, test_loss = test_with_roc([testing_input, testing_labels, testing_names], model, loss)

    print("Test Accuracy: " + str(test_accu) + "%")


class Main():
    def __init__(self):
        warnings.filterwarnings("ignore")
        print("Hello!")
        self.question = input("Build new model? : ")

        if self.question == "yes" or self.question == "Yes":
            build_model()
            self.question = input("Test existing model? : ")

            if self.question == "yes" or self.question == "Yes":
                test_model()
        else:
            self.question = input("Test existing model? : ")

            if self.question == "yes" or self.question == "Yes":
                test_model()

        print("FINISHED")


# Good Luck!
main = Main()
