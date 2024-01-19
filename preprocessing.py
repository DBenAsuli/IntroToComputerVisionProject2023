# Intro to Computer Vision             Project
# Dvir Ben Asuli                       318208816
# The Open University                  January 2023

# Intro to Computer Vision             Project
# Dvir Ben Asuli                       318208816
# The Open University                  January 2023

from common import *

import cv2
import h5py
import random
import numpy as np
from keras.utils import np_utils

DATABASE_PATH = "./data/"
IMAGES_PATH = "./images"
DATABASE_NAME = "SynthText_train.h5"
IMG_SIZE = 64
NUM_OF_SAMLES = 30520


# Importing the Database from the h5py file
def import_db(file_name):
    db = h5py.File(file_name, 'r')
    im_names = list(db['data'].keys())

    return db, im_names


# Extracting values from imported database
def read_dataset(db, im_names, index):
    im = im_names[index]
    img = db['data'][im][:]
    font = db['data'][im].attrs['font']
    txt = db['data'][im].attrs['txt']
    charBB = db['data'][im].attrs['charBB']
    wordBB = db['data'][im].attrs['wordBB']

    return [img, font, txt, charBB, wordBB]


# Taking a picture and extracting windows of all the characters, with
# their corresponding font label
def extract_letters(img, font, txt, charBB):
    chars_dataset = []
    # im.show()

    for letter in range(charBB.shape[2]):
        max_x = max(max(charBB[0, :, letter]), 0)
        min_x = max(min(charBB[0, :, letter]), 0)
        max_y = max(max(charBB[1, :, letter]), 0)
        min_y = max(min(charBB[1, :, letter]), 0)

        char_img = img[int(min_y): int(max_y), int(min_x): int(max_x)]
        char_img = cv2.resize(char_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        label = extract_labels(font[letter].decode())

        chars_dataset.append([char_img, label])

    return chars_dataset


# Taking a picture and extracting windows of all the characters, with
# their corresponding font label and original image name
def extract_letters_with_names(img, font, txt, charBB, img_name):
    chars_dataset = []
    # im.show()

    for letter in range(charBB.shape[2]):
        max_x = max(max(charBB[0, :, letter]), 0)
        min_x = max(min(charBB[0, :, letter]), 0)
        max_y = max(max(charBB[1, :, letter]), 0)
        min_y = max(min(charBB[1, :, letter]), 0)

        char_img = img[int(min_y): int(max_y), int(min_x): int(max_x)]
        char_img = cv2.resize(char_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        label = extract_labels(font[letter].decode())

        chars_dataset.append([char_img, label, img_name])

    return chars_dataset


# Building the complete dataabase of the training and validation process
# Taking the database and building a list of pairs of [char_img, font_label]
def build_complete_dataset(path):
    database = []
    dataset = []
    db, im_names = import_db(path)

    for i in range(0, 998):
        database.append(read_dataset(db, im_names, i))

    for i in range(0, 998):
        temp_dataset = extract_letters(database[i][0], database[i][1], database[i][2], database[i][3])
        dataset = dataset + temp_dataset

    random.shuffle(dataset)
    return dataset


# Building the complete dataabase of the training and validation process
# Taking the database and building a list of pairs of [char_img, font_label, original_img_name]
def build_complete_dataset_with_names(path):
    database = []
    dataset = []
    db, im_names = import_db(path)

    for i in range(0, 998):
        database.append(read_dataset(db, im_names, i))
        database[i].append(im_names[i].replace(".jpg_0", "").replace(".jpeg_0", ""))

    for i in range(0, 998):
        temp_dataset = extract_letters_with_names(database[i][0], database[i][1], database[i][2], database[i][3],
                                                  database[i][5])
        dataset = dataset + temp_dataset

    random.shuffle(dataset)
    return dataset


# Converting a text label of font to a number
def extract_labels(label_txt):
    if label_txt == 'Alex Brush':
        return 0
    if label_txt == 'Open Sans':
        return 1
    if label_txt == 'Sansation':
        return 2
    if label_txt == 'Ubuntu Mono':
        return 3
    if label_txt == 'Titillium Web':
        return 4


# Building windows in length 20 of [char_img, font_label] pairs
def build_model_windows(dataset):
    i = 0
    j = 0

    random.shuffle(dataset)

    temp_res = []
    res = []
    while True:
        temp_item = []
        temp_item.append(dataset[j][0])
        temp_item.append(dataset[j][1])
        temp_res.append(temp_item)

        if i == WINDOW_SIZE - 1:
            res.append(temp_res)
            temp_res = []
            i = 0
        else:
            i = i + 1

        if j == NUM_OF_SAMLES - 1:
            break
        else:
            j = j + 1

    return res


def build_model_input(set):
    X = []
    Y_temp = []
    for features, label in set:
        features = np.array(features).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        X.append(features)
        Y_temp.append(label)

    Y = np_utils.to_categorical(Y_temp, 5)
    return X, Y


def build_model_input_with_names(set):
    X = []
    Y_temp = []
    Names = []
    for features, label, name in set:
        features = np.array(features).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        X.append(features)
        Y_temp.append(label)
        Names.append(name)

    Y = np_utils.to_categorical(Y_temp, 5)
    return X, Y, Names
