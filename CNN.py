# Intro to Computer Vision             Project
# Dvir Ben Asuli                       318208816
# The Open University                  January 2023

from common import *

import torch
import torchvision
import torch.nn as nn

OUTPUT_SIZE = 738048
KERNEL_SIZE = (3, 3)


class CNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=KERNEL_SIZE, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * in_channels, out_channels=4 * in_channels, kernel_size=KERNEL_SIZE, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * in_channels, out_channels=8 * in_channels, kernel_size=KERNEL_SIZE, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * in_channels, out_channels=16 * in_channels, kernel_size=KERNEL_SIZE, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * in_channels, out_channels=32 * in_channels, kernel_size=KERNEL_SIZE, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 * in_channels, out_channels=64 * in_channels, kernel_size=KERNEL_SIZE, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64 * in_channels, out_channels=64 * in_channels, kernel_size=KERNEL_SIZE)

        )
        self.classifier = nn.Sequential(
            nn.Linear(OUTPUT_SIZE, 5),
            nn.Softmax()

        )

    def forward(self, features):
        features = self.feature_extractor(features)
        features = features.view(features.size(0), -1)
        features = torch.flatten(features, start_dim=1)
        class_scores = self.classifier(features)
        return class_scores


# Training the Network
def train(test_data, train_data, net, optimizer, criterion, num_epochs=120):
    train_accu = []
    train_loss = []
    test_loss = []
    test_accu = []

    net.train()
    trans = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    for epoch_idx in range(num_epochs):

        total_loss = 0
        correct = 0

        for i in range(int(len(train_data[0]))):
            X = torch.from_numpy(train_data[0][i]).float()
            X = trans(X.transpose(1, 3))
            y = torch.from_numpy(train_data[1][i]).float()

            # Forward pass
            if len(X) != 0:
                optimizer.zero_grad()  # Zero gradients of all parameters
                y_pred = net(X)
                # Compute loss
                loss = criterion(y_pred[0], y)
                total_loss += loss.item()

                # Backward pass
                loss.backward()  # Run backprop algorithms to calculate gradients

                # Optimization step
                optimizer.step()  # Use gradients to update model parameters

                # Getting correct count
                for j in range(len(y.flatten())):
                    if y.flatten()[j] == 1:
                        max_value = max(y_pred[0].flatten())
                        index = y_pred[0].flatten().tolist().index(max_value)
                        if index == j:
                            correct += 1

        accu = 100 * (correct / int(len(train_data[0])))
        acc_test, loss_test = test(test_data, net, criterion)

        train_loss.append(total_loss)
        train_accu.append(accu)
        test_accu.append(acc_test)
        test_loss.append(loss_test)
        print(f"End of epoch: {epoch_idx}\nTrain loss: {total_loss}\nTrain accuracy: {accu} ")
        print(f"\nTest loss: {loss_test}\nTest accuracy: {acc_test} \n")

    return train_loss, test_loss, train_accu, test_accu


# Testing the Network
def test(test_data, net, criterion):
    total_loss = 0
    correct = 0

    net.eval()
    with torch.no_grad():
        for i in range(int(len(test_data[0]))):
            X = torch.from_numpy(test_data[0][i]).float()
            y = torch.from_numpy(test_data[1][i]).float()

            if len(X) != 0:
                # Forward Pass
                y_pred = net(X.transpose(1, 3))

                # Compute loss
                loss = criterion(y_pred[0], y)
                total_loss += loss.item()

                # Getting correct count
                for j in range(len(y.flatten())):
                    if y.flatten()[j] == 1:
                        max_value = max(y_pred[0].flatten())
                        index = y_pred[0].flatten().tolist().index(max_value)
                        if index == j:
                            correct += 1

    accu = 100 * (correct / int(len(test_data[0])))

    return accu, total_loss


def test_with_roc(test_data, net, criterion):
    total_loss = 0
    correct = 0
    complete_lables = []
    complete_predictions = []
    results_dict = {}
    fonts = ["Alex", "Open-Sans", "Sensations", "Ubunto-Mono", "Titilium"]

    net.eval()
    with torch.no_grad():
        for i in range(int(len(test_data[0]))):
            X = torch.from_numpy(test_data[0][i]).float()
            y = torch.from_numpy(test_data[1][i]).float()
            img_name = test_data[2][i]

            if img_name not in results_dict.keys():
                results_dict[img_name] = ()

            if len(X) != 0:

                # Forward Pass
                y_pred = net(X.transpose(1, 3))

                # Accumulating all labels and predictions
                complete_predictions.append(y_pred[0].flatten())
                complete_lables.append(y.flatten().flatten())

                # Compute loss
                loss = criterion(y_pred[0], y)
                total_loss += loss.item()

                # Getting correct count
                for j in range(len(y.flatten())):
                    if y.flatten()[j] == 1:
                        max_value = max(y_pred[0].flatten())
                        index = y_pred[0].flatten().tolist().index(max_value)
                        if index == j:
                            if fonts[j] not in results_dict[img_name]:
                                # Updating the fonts list of an image
                                results_dict[img_name] = results_dict[img_name] + (fonts[j],)
                            correct += 1

    accu = 100 * (correct / int(len(test_data[0])))

    generate_roc(np.array(complete_lables, dtype=object), np.array(complete_predictions, dtype=object))
    print_results(results_dict)

    return accu, total_loss
