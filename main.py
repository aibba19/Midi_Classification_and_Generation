from transformer_classifier import TransformerClassifier

import os
import dataset_creation
import sys

sys.path.append(
    r'C:\Users\andri\Desktop\Tesi\prove_classificazione\midi_classification copy\tegridy-tools\tegridy-tools')

base_dir = os.getcwd()

from GPT2RGAX import *

import pandas as pd

import numpy as np

from lstm import LSTM

from preprocess_data import process_melody_chords

def nn_configurations():
    configs = [
        {'epochs':250, 'neurons':64, 'batch_size':256},
        {'epochs':250, 'neurons':128, 'batch_size':256},
        {'epochs':500, 'neurons':128, 'batch_size':512},
        {'epochs':1000, 'neurons':64, 'batch_size':1024},
        {'epochs':1000, 'neurons':128, 'batch_size':1204},
    ]
    for config in configs:
        yield config

def train_model():

    train_list_x, train_list_y = process_melody_chords()

    #train_list_x = train_list_x.reshape(train_list_x.shape[0], 50, 1)
    #train_list_y = train_list_y.reshape(train_list_y.shape[0], 5)

    print('Done!')
    print('=' * 70)

    vocab_size = np.amax(train_list_x)
    feed_shape = np.shape(train_list_x[0])[0]

    print('Total lists of 50 events each:', len(train_list_x))
    print('Feed Shape:', feed_shape)
    print('Maximum INT:', vocab_size - 1)
    print('Unique INTs:', len(np.unique(train_list_x)))
    #print('Intro/Zero INTs:', train_list_x.count(0))
    print('=' * 70)

    #X_train, y_train, X_test, y_test = train_test_split(train_list_x, train_list_y, test_size=0.2)
    #X_train, y_train, X_val, y_val = train_test_split(X_train, y_train, test_size=0.25)

    X_train = train_list_x[:int(len(train_list_x) * 0.70)]
    y_train = train_list_y[:int(len(train_list_y) * 0.70)]

    X_val = train_list_x[int(len(train_list_x) * 0.70):int(len(train_list_x) * 0.85)]
    y_val = train_list_y[int(len(train_list_y) * 0.70):int(len(train_list_y) * 0.85)]

    X_test = train_list_x[int(len(train_list_x) * 0.85):]
    y_test = train_list_y[int(len(train_list_y) * 0.85):]

    config = {'epochs': 250, 'neurons': 64, 'batch_size': 16}

    model = TransformerClassifier(feed_shape, vocabulary_size=vocab_size, config=config)
    #model = LSTM(input_shape=feed_shape, vocabulary_size=vocab_size)

    model.fit(X_train, X_val, y_train, y_val)
    metrics = model.model.evaluate(X_test, y_test)


    #model.summary()
    accuracies = []
    results = []
    accuracies.append(metrics)

    #results['avg_loss'] = sum([l[0] for l in accuracies]) / len(accuracies) if not len(accuracies) == 0 else 0
    #results['avg_accuracy'] = sum([l[1] for l in accuracies]) / len(accuracies) if not len(accuracies) == 0 else 0


# train data = list of list of int, where each list is a song
# Prepare data in numpy format to feed the lstm nn
def get_train_val_split(train_data1):
    # Divide the dataset in train, validation e test set
    train_data = train_data1[:int(len(train_data1))]

    # Val and test set are the same here
    val_dataset = train_data[:int(len(train_data) * val_dataset_ratio)]
    # Non capisco perch+ il test db è uguale al validation
    test_dataset = train_data[:int(len(train_data) * val_dataset_ratio)]

    train_list = train_data
    val_list = val_dataset
    test_list = []
    print('=' * 50)

    # prendo le label dal csv dove le ho annotate
    train_data_csv = pd.read_csv(
        r"C:\Users\andri\Desktop\Tesi\prove_classificazione\midi_classification copy\vgmidi_labelled.csv", sep=";")
    # print("example ", train_data_csv.loc1)
    # Notare che c'era un errore e label ha uno spazio
    train_data_y = train_data_csv['label '].tolist()

    # Create the df with array of train and labels
    df = pd.DataFrame()
    df["x_train"] = pd.Series(train_data1)
    df["y_train"] = train_data_y
    df.to_csv('dataset_labelled.csv', index=False, sep=';', encoding='utf-8')

    print("Len di label: ", len(train_data_y))

    vocab_size = max(np.max(df["x_train"])) + 1

    print("Vocab_size : ", vocab_size)

    # model = lstm(input_shape=input_size, vocabulary_size=vocab_size)

    # Create ndarray for feed the network
    x_train = np.array([np.array(xi) for xi in train_list])
    x_val = np.array([np.array(xi) for xi in val_list])

    y_train = np.asarray(train[1]).astype('int')
    y_val = np.asarray(val[1]).astype('int')

    x_test = np.asarray(test[0]).astype('int')
    y_test = np.asarray(test[1]).astype('int')


if __name__ == "__main__":

    #dataset_creation.create_nes_dataset()

    print('Creating dataset dirs...')

    if not os.path.exists(os.path.join(base_dir, 'ints_dataset')):
        os.mkdir('ints_dataset')

    print('Ints Dataset folder created!')

    os.chdir(r'C:\Users\andri\Desktop\Tesi\prove_classificazione\midi_classification copy\tegridy-tools\tegridy-tools')
    os.chdir(base_dir)

    print("We are inside this folder" + str(os.path.abspath(os.getcwd())))
    print('Loading TMIDIX module...')

    train_model()









