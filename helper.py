# -*- encoding:utf-8 -*-
import os
import json
import pickle

import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

TRAIN_PICKLE = "sequences.train.pickle"
TEST_PICKLE = "sequences.test.pickle"
DEV_PICKLE = "sequences.dev.pickle"
TOKENIZER_PICKLE = "tokenizer.pickle"

NB_CLASSES = 3
label2class = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

def _formatting(line):
    row = json.loads(line)
    x_1 = row['sentence1']
    x_2 = row['sentence2']
    y = label2class[row['gold_label']]
    return x_1, x_2, y


def load_data(debug=False):
    if (os.path.exists(TRAIN_PICKLE)
        and os.path.exists(TEST_PICKLE)
        and os.path.exists(DEV_PICKLE)):

        with open(TRAIN_PICKLE, 'rb') as fp:
            X_train_1, X_train_2, Y_train = pickle.load(fp)
        with open(TEST_PICKLE, 'rb') as fp:
            X_test_1, X_test_2, Y_test = pickle.load(fp)
        with open(DEV_PICKLE, 'rb') as fp:
            X_dev_1, X_dev_2, Y_dev = pickle.load(fp)

    else:
        x_train_1, x_train_2, y_train = [], [], []
        x_test_1, x_test_2, y_test = [], [], []
        x_dev_1, x_dev_2, y_dev = [], [], []

        with open("snli_1.0_train.jsonl", encoding='utf8') as fp:
            for line in fp:
                try:
                    x_1, x_2, y = _formatting(line)
                    x_train_1.append(x_1)
                    x_train_2.append(x_2)
                    y_train.append(y)
                except KeyError:
                    continue

        with open("snli_1.0_test.jsonl", encoding='utf8') as fp:
            for line in fp:
                try:
                    x_1, x_2, y = _formatting(line)
                    x_test_1.append(x_1)
                    x_test_2.append(x_2)
                    y_test.append(y)
                except KeyError:
                    continue

        with open("snli_1.0_dev.jsonl", encoding='utf8') as fp:
            for line in fp:
                try:
                    x_1, x_2, y = _formatting(line)
                    x_dev_1.append(x_1)
                    x_dev_2.append(x_2)
                    y_dev.append(y)
                except KeyError:
                    continue

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x_train_1)
        tokenizer.fit_on_texts(x_train_2)
        tokenizer.fit_on_texts(x_test_1)
        tokenizer.fit_on_texts(x_test_2)
        tokenizer.fit_on_texts(x_dev_1)
        tokenizer.fit_on_texts(x_dev_2)

        X_train_1 = tokenizer.texts_to_sequences(x_train_1)
        X_train_2 = tokenizer.texts_to_sequences(x_train_2)
        X_test_1 = tokenizer.texts_to_sequences(x_test_1)
        X_test_2 = tokenizer.texts_to_sequences(x_test_2)
        X_dev_1 = tokenizer.texts_to_sequences(x_dev_1)
        X_dev_2 = tokenizer.texts_to_sequences(x_dev_2)

        MAX_SEQUENCE_LENGTH = max([len(seq) for seq in X_train_1 + X_train_2
                                                     + X_test_1 + X_test_2
                                                     + X_dev_1 + X_dev_2])
        # print(X_train_1 + X_train_2 + X_test_1 + X_test_2 + X_dev_1 + X_dev_2)
        MAX_NB_WORDS = len(tokenizer.word_index) + 1

        if debug:
            print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
            print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))

        X_train_1 = pad_sequences(X_train_1, maxlen=MAX_SEQUENCE_LENGTH)
        X_train_2 = pad_sequences(X_train_2, maxlen=MAX_SEQUENCE_LENGTH)
        X_test_1 = pad_sequences(X_test_1, maxlen=MAX_SEQUENCE_LENGTH)
        X_test_2 = pad_sequences(X_test_2, maxlen=MAX_SEQUENCE_LENGTH)
        X_dev_1 = pad_sequences(X_dev_1, maxlen=MAX_SEQUENCE_LENGTH)
        X_dev_2 = pad_sequences(X_dev_2, maxlen=MAX_SEQUENCE_LENGTH)

        Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
        Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
        Y_dev = np_utils.to_categorical(y_dev, NB_CLASSES)

        with open(TRAIN_PICKLE, 'wb') as fp:
            pickle.dump((X_train_1, X_train_2, Y_train), fp)
        with open(TEST_PICKLE, 'wb') as fp:
            pickle.dump((X_test_1, X_test_2, Y_test), fp)
        with open(DEV_PICKLE, 'wb') as fp:
            pickle.dump((X_dev_1, X_dev_2, Y_dev), fp)

        with open(TOKENIZER_PICKLE, 'wb') as fp:
            pickle.dump(tokenizer, fp)

    return (X_train_1, X_train_2, Y_train,
            X_test_1, X_test_2, Y_test,
            X_dev_1, X_dev_2, Y_dev)


def load_tokenizer():
    if not os.path.exists(TOKENIZER_PICKLE):
        load_data()
    with open(TOKENIZER_PICKLE, 'rb') as fp:
        tokenizer = pickle.load(fp)
    return tokenizer
