# -*- encoding:utf-8 -*-
import os
import pickle

import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.models import load_model as K_load_model
from keras.layers import Input, Embedding, LSTM, Dense, Merge, Activation

from helper import load_data, load_tokenizer

EMBEDDING_DIM = 300
LSTM_DIM = 256
MAX_SEQUENCE_LENGTH = 78
MAX_NB_WORDS = 34873

OPTIMIZER = 'adagrad'
BATCH_SIZE = 256
NB_EPOCH = 10

TRAINED_CLASSIFIER_PATH = "dual_encoder_lstm_classifier.h5"
GLOVE_INDEX_PICKLE = "glove_index.pickle"

def load_embedding_matrix():
    embeddings_index = {}

    print("\tNow loading GloVe embedding...")
    if os.path.exists(GLOVE_INDEX_PICKLE):
        with open(GLOVE_INDEX_PICKLE, 'rb') as fp:
            embeddings_index = pickle.load(fp)
    else:
        with open("glove.6B.{}d.txt".format(EMBEDDING_DIM), encoding='utf8') as fp:
            for line in fp:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            print("Found {} word vectors.".format(embeddings_index))

        with open(GLOVE_INDEX_PICKLE, 'wb') as fp:
            pickle.dump(embeddings_index, fp)

    empty_embedding_count = 0

    tokenizer = load_tokenizer()
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector
        else:
            empty_embedding_count += 1

    print("Empty embedding words: {}".format(empty_embedding_count))
    return embedding_matrix

def load_model():
    if not os.path.exists(TRAINED_CLASSIFIER_PATH):
        print("No pre-trained model...")
        print("Start building model...")

        print("Now loading SNLI data...")
        X_train_1, X_train_2, Y_train, X_test_1, X_test_2, Y_test, X_dev_1, X_dev_2, Y_dev = load_data()

        print("Now loading embedding matrix...")
        embedding_matrix = load_embedding_matrix()

        print("Now building dual encoder lstm model...")
        # define lstm for sentence1
        branch1 = Sequential()
        branch1.add(Embedding(output_dim=EMBEDDING_DIM,
                              input_dim=MAX_NB_WORDS,
                              input_length=MAX_SEQUENCE_LENGTH,
                              weights=[embedding_matrix],
                              mask_zero=True,
                              trainable=False))
        branch1.add(LSTM(output_dim=LSTM_DIM))

        # define lstm for sentence2
        branch2 = Sequential()
        branch2.add(Embedding(output_dim=EMBEDDING_DIM,
                              input_dim=MAX_NB_WORDS,
                              input_length=MAX_SEQUENCE_LENGTH,
                              weights=[embedding_matrix],
                              mask_zero=True,
                              trainable=False))
        branch2.add(LSTM(output_dim=LSTM_DIM))

        # define classifier model
        model = Sequential()
        # Merge layer holds a weight matrix of itself
        model.add(Merge([branch1, branch2], mode='mul'))
        model.add(Dense(3))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=OPTIMIZER,
                      metrics=['accuracy'])

        print("Now training the model...")
        print("\tbatch_size={}, nb_epoch={}".format(BATCH_SIZE, NB_EPOCH))
        model.fit([X_train_1, X_train_2], Y_train,
                  batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
                  validation_data=([X_test_1, X_test_2], Y_test))

        print("Now saving the model... at {}".format(TRAINED_CLASSIFIER_PATH))
        model.save(TRAINED_CLASSIFIER_PATH)

    else:
        print("Found pre-trained model...")
        model = K_load_model(TRAINED_CLASSIFIER_PATH)

    return model

if __name__ == "__main__":
    model = load_model()
