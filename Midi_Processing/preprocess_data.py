import json
from random import random

from tensorflow import keras

print('Loading needed modules. Please wait...')
import os
import tqdm as tqdm
from tqdm import tqdm

import sys

sys.path.append(
    r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\libraries\tegridy-tools\tegridy-tools')

base_dir = os.getcwd()

import TMIDIX

#from GPT2RGAX import *

import pandas as pd

import numpy as np
from sklearn import preprocessing

from transformers import AutoTokenizer

with open(os.path.join(base_dir, "../Classification/lists/rpg.json"), "r") as fp:
    rpg = json.load(fp)
with open(os.path.join(base_dir, "../Classification/lists/fighting.json"), "r") as fp:
    fighting = json.load(fp)
with open(os.path.join(base_dir, "../Classification/lists/puzzle.json"), "r") as fp:
    puzzle = json.load(fp)
with open(os.path.join(base_dir, "../Classification/lists/sport.json"), "r") as fp:
    sport = json.load(fp)
with open(os.path.join(base_dir, "../Classification/lists/shooting.json"), "r") as fp:
    shooting = json.load(fp)

label_lists = rpg + fighting + puzzle + sport + shooting

def create_melody_chords_dataset(target_db):
    # Process MIDIs to special MIDI dataset with TMIDIX MIDI Processor

    sorted_or_random_file_loading_order = True  # Sorted order is NOT recommended
    dataset_ratio = 1  # Change this if you need more data

    print('TMIDIX MIDI Processor')
    print('Starting up...')
    ###########

    files_count = 0

    gfiles = []

    melody_chords_f = []

    ###########

    print('Loading MIDI files...')
    print('This may take a while on a large dataset in particular.')

    # Qui ci sono le canzoni di vgmidi db
    #dataset_addr = r"C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\dataset\vgmidi_training"

    if target_db == 'Classification':
        # Qui ci sono le canzoni relative al nes_db che sono state etichettate nei diversi generi del gioco di riferimento
        # quindi quelle che servono per il training del classificatore
        dataset_addr = r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\dataset\nes_training'
        pickle_path = '../Classification/chords_dataset'
        csv_filename = 'melody_chords_classification'

    else:
        # Qui sono presenti tutte le canzoni del nes db
        dataset_addr = r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\dataset\nes\nes'
        pickle_path = '../Generation/chords_dataset'
        csv_filename = 'melody_chords_generation'

    # os.chdir(dataset_addr)
    filez = list()
    songnames = list()
    for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
        filez += [os.path.join(dirpath, file) for file in filenames]

        # Creo array con i titoli delle canzoni
        songnames = filenames
        print("Title's songs in the dataset")
        print(songnames)

    print('=' * 70)

    if filez == []:
        print('Could not find any MIDI files. Please check Dataset dir...')
        print('=' * 70)

    if sorted_or_random_file_loading_order:
        print('Sorting files...')
        filez.sort()
        print('Done!')
        print('=' * 70)
    else:
        print('Randomizing file list...')
        random.shuffle(filez)

    stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    print('Processing MIDI files. Please wait...')
    for f in tqdm(filez[:int(len(filez) * dataset_ratio)]):
        try:
            # get the path of the file
            fn = os.path.basename(f)
            # get the filename
            fn1 = fn.split('.')[0]

            files_count += 1

            # print('Loading MIDI file...')
            score = TMIDIX.midi2ms_score(open(f, 'rb').read())

            events_matrix = []

            itrack = 1

            patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            patch_map = [[0, 1, 2, 3, 4, 5, 6, 7],  # Piano
                         [24, 25, 26, 27, 28, 29, 30],  # Guitar
                         [32, 33, 34, 35, 36, 37, 38, 39],  # Bass
                         [40, 41],  # Violin
                         [42, 43],  # Cello
                         [46],  # Harp
                         [56, 57, 58, 59, 60],  # Trumpet
                         [71, 72],  # Clarinet
                         [73, 74, 75],  # Flute
                         [-1],  # Fake Drums
                         [52, 53]  # Choir
                         ]

            while itrack < len(score):
                for event in score[itrack]:
                    if event[0] == 'note' or event[0] == 'patch_change':
                        events_matrix.append(event)
                itrack += 1

            events_matrix1 = []
            for event in events_matrix:
                if event[0] == 'patch_change':
                    patches[event[2]] = event[3]

                if event[0] == 'note':
                    event.extend([patches[event[3]]])
                    once = False

                    for p in patch_map:
                        if event[6] in p and event[3] != 9:  # Except the drums
                            event[3] = patch_map.index(p)
                            once = True

                    if not once and event[3] != 9:  # Except the drums
                        event[3] = 0  # All other instruments/patches channel
                        event[5] = max(80, event[5])

                    if event[3] < 11:  # We won't write chans 11-16 for now...
                        events_matrix1.append(event)
                        stats[event[3]] += 1

            # recalculating timings yoda version
            for e in events_matrix1:
                e[1] = int(e[1] / 16)
                e[2] = int(e[2] / 128)

            # final processing...

            # =======================

            if len(events_matrix1) > 0:
                events_matrix1.sort(key=lambda x: (x[1], x[4]))

                cho = []
                pe = events_matrix1[0]
                melody_chords = []
                # l = []
                for e in events_matrix1:
                    # Yoda Version
                    time = max(0, min(255, e[1] - pe[1]))
                    dur = max(0, min(15, e[2]))
                    cha = max(0, min(15, e[3]))
                    ptc = max(0, min(127, e[4]))
                    vel = max(0, min(127, e[5]))

                    # This contain the melody chords for the song that is in processing
                    melody_chords.append([time, dur, ptc, cha, vel])

                    pe = e

                # This variable contains all the melody chords for all files
                melody_chords_f.append(melody_chords)

            # this list contains the path to all midi files used in the process
            gfiles.append(f)

        except KeyboardInterrupt:
            print('Saving current progress and quitting...')
            break

        except:
            print('Bad MIDI:', f)
            continue

    # Create df with melody chords
    create_df(songnames, melody_chords_f, csv_filename)

    print('=' * 70)

    print('Done!')
    print('=' * 70)

    print('Resulting Stats:')
    print('=' * 70)

    print('Piano:', stats[0])
    print('Guitar:', stats[1])
    print('Bass:', stats[2])
    print('Violin:', stats[3])
    print('Cello:', stats[4])
    print('Harp:', stats[5])
    print('Trumpet:', stats[6])
    print('Clarinet:', stats[7])
    print('Flute:', stats[8])
    print('Drums:', stats[9])
    print('Choir:', stats[10])

    print('=' * 70)

    # More dataset stats...

    times = []
    durs = []
    pitches = []

    for chords_list in tqdm(melody_chords_f):
        for c in chords_list:
            times.append(c[0])
            durs.append(c[1])
            pitches.append(c[2])

    tavg = sum(times) / len(times)
    davg = sum(durs) / len(durs)
    pavg = sum(pitches) / len(pitches)

    print('Average time-shift', tavg)
    print('Average duration', davg)
    print('Average pitch', pavg)
    print('=' * 70)

    # Write the processed db in a pickle file to be load lately for gurther processing
    TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, pickle_path)


def process_melody_chords(task):
    # Yoda Vers 2.0 Process and prep INTs...
    randomize_dataset = True

    if task == 'classification':
        if not os.path.exists('../Classification/chords_dataset.pickle'):
            create_melody_chords_dataset('Classification')

        melody_chords_f = TMIDIX.Tegridy_Any_Pickle_File_Reader('../Classification/chords_dataset')

        # if randomize_dataset:
        #     print('=' * 70)
        #     print('Randomizing the dataset...')
        #     random.shuffle(melody_chords_f)
        #     print('Done!')

        train_data_y = load_nes_label()
        train_list_x, train_list_y = build_list_of_max_n_tokens(melody_chords_f, train_data_y)

        return train_list_x, train_list_y

    else:
        if not os.path.exists('../Generation/chords_dataset.pickle'):
            create_melody_chords_dataset('Generation')

        melody_chords_f = TMIDIX.Tegridy_Any_Pickle_File_Reader('../Generation/chords_dataset')

        train_list_x = build_one_stream_list_for_finetuning(melody_chords_f)
        return train_list_x


    #Different approach to create the target label values
    #train_data_y = load_vgmidi_label()
    #train_data_y = load_nes_label()
    #create_nes_label()

    #Two different options for creating the train database
    #train_list_x, train_list_y = build_list_of_max_50_tokens(melody_chords_f,train_data_y)

    #train_list_x, train_list_y = build_one_list_for_each_song(melody_chords_f, train_data_y)
    #train_list_x, train_list_y = build_list_of_max_n_tokens(melody_chords_f, train_data_y)

    # df = pd.DataFrame()
    # df['song_token_list'] = train_list_x
    # df['label'] = train_data_y
    # df.to_csv('token_list_and_label.csv', index=False, sep=';')

    # posso usare questa funzione per trasformare le liste da liste a stringhe
    #''.join(map(str, train_list_x[0]))

    # PROVA DA COMPLETARE
    # trasforma ogni lista di token riferita ad una singola canzone in stringa
    # per poter utilizzare un tokenizzatore per preparare i dati per la rete neurale
    #Da modificare il fatto che si inverte l'ordine delle liste
    # X = []
    # for tokens in train_list_x:
    #     v = ' '.join(map(str, tokens))
    #
    #     X.append(v)

    # Prova di tokenizzazione trainandolo da zero perch?? il mio ?? un dominio particolare


    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import BertProcessing
    # from pathlib import Path
    #
    # paths = [str(x) for x in Path(".").glob("text_split/*.txt")]

    # Initialize a tokenizer
    #tokenizer = ByteLevelBPETokenizer(lowercase=True)

    # Customize training
    # tokenizer.train(files=paths, vocab_size=8192, min_frequency=2,
    #                 show_progress=True,
    #                 special_tokens=[
    #                     "<s>",
    #                     "<pad>",
    #                     "</s>",
    #                     "<unk>",
    #                     "<mask>",
    #                 ])
    # Save the Tokenizer to disk
    #tokenizer.save_model(tokenizer_folder)

    #new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=25000)

    # for i, phrase in enumerate(df['Phrase']):
    #     tokens = tokenizer.encode_plus(phrase, max_length=seq_len, truncation=True,
    #                                    padding='max_length', add_special_tokens=True,
    #                                    return_tensors='tf')
    #     # assign tokenized outputs to respective rows in numpy arrays
    #     Xids[i, :] = tokens['input_ids']
    #     Xmask[i, :] = tokens['attention_mask']


def load_vgmidi_label():
    # prendo le label dal csv dove le ho annotate
    train_data_csv = pd.read_csv(
        r"C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\vgmidi_labelled.csv", sep=";")

    train_data_y = train_data_csv['label1'].tolist()

    return train_data_y


def load_nes_label():
    # prendo le label dal csv dove le ho annotate
    #train_data_csv = pd.read_csv(
    #    r"C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\nes_labelled.csv", sep=";")

    train_data_csv = pd.read_csv(r"../Classification/nes_labelled.csv",
                                 sep=";", converters={'label': eval})

    train_data_y = train_data_csv['label'].tolist()

    return train_data_y


def build_one_stream_list_for_finetuning(melody_chords_f):

    # randomize_dataset = True

    #
    # if randomize_dataset:
    #     print('=' * 70)
    #     print('Randomizing the dataset...')
    #     random.shuffle(melody_chords_f)
    #     print('Done!')

    train_list_x = []

    # Cycle that build a list for every song
    for chords_list in tqdm(melody_chords_f):

        train_list_x.extend([0])  # Intro/Zero Token

        for i in chords_list:

            if i[0] != 0:  # This is the chordification line
                train_list_x.extend([i[0]])  # start-times

            # And this is the main MIDI note line (triple stack)
            main_note = [i[1] + (i[2] * 16) + (i[3] * 16 * 128)]  # Main note == [duration / pitch / channel]

            if main_note != [0]:  # Main note error control...
                train_list_x.extend(main_note)  # Main note == [duration / pitch / channel]

    print('=' * 70)

    print('Total INTs:', len(train_list_x))
    print('Minimum INT:', min(train_list_x))
    print('Maximum INT:', max(train_list_x))
    print('Unique INTs:', len(set(train_list_x)))
    print('Intro/Zero INTs:', train_list_x.count(0))
    print('=' * 70)

    return train_list_x

def build_one_list_for_each_song(melody_chords_f,train_data_y):

    train_list_x = []
    train_list_y = []
    song_number = 0

    # Cycle that build a list for every song
    for chords_list in tqdm(melody_chords_f):

        train_list_x.append([0])  # Intro/Zero Token

        for i in chords_list:

            if i[0] != 0:  # This is the chordification line
                train_list_x[song_number].extend([i[0]])  # start-times

            # And this is the main MIDI note line (triple stack)
            main_note = [i[1] + (i[2] * 16) + (i[3] * 16 * 128)]  # Main note == [duration / pitch / channel]

            if main_note != [0]:  # Main note error control...
                train_list_x[song_number].extend(main_note)  # Main note == [duration / pitch / channel]

        train_list_y.append(train_data_y[song_number])

        song_number = song_number + 1

    #Al posto di questo proviamo a creare un tokenizer utilizzando la lista di interi come un nuovo linguaggio

    # Potrebbe essere una criticit?? il fatto di fare il padding con 0
    # DA VERIFICARE
    # train_list_x = list_padding(train_list_x, 0)
    #
    # train_list_x = [np.array(x) for x in train_list_x]
    #train_list_x = np.array(train_list_x)
    #
    # train_list_y = np.array(train_list_y)
    #
    train_list_y = one_hot_encoding(train_list_y)
    #
    # return train_list_x, train_list_y


    #maxlen = 2000

    train_list_x = np.array(train_list_x)
    train_list_y = np.array(train_list_y)

    train_list_x = keras.preprocessing.sequence.pad_sequences(train_list_x, padding='post')

    return train_list_x, train_list_y

def train_tokenizer_from_scratch(train_list):
    # ! pip install tokenizers

    from pathlib import Path

    from tokenizers import ByteLevelBPETokenizer

    paths = [str(x) for x in train_list]
    #paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<0>"
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save_model(".", r"C:\Users\andri\Desktop\Tesi\Progetto\tokenizer\midi_tokenizer")


def build_list_of_max_n_tokens(melody_chords_f, train_data_y):

    # Funzionante
    SEQ_LEN = 75
    BATCH_SIZE = 16

    train_list_x = []
    train_list_y = []
    song_number = 0

    # This cycle divide the song token in list of max 50 token lenght
    # Each chords list represent a song
    #[time, dur, ptc, cha, vel]
    for chords_list in tqdm(melody_chords_f):

        # train_data1.extend([0])  # Intro/Zero Token
        # Divide le tracce in j parti da 50 eventi di chord list
        if len(chords_list) < SEQ_LEN:
            r = 1
        else:
            r = int(len(chords_list) / SEQ_LEN) + 1
        for j in range(r):
            # If we are at the first slice of the song we add the 0 intro token
            if (j == 0):
                curr_sample = [0]
            else:
                curr_sample = []
            # curr_sample = []
            sample_to_process = chords_list[(j * SEQ_LEN):((j + 1) * SEQ_LEN)]
            # Slicing on chord list to take only 50 event max
            # Note that for each event in curr_sample could add two int in curr sample
            for i in sample_to_process:

                if i[0] != 0:  # This is the chordification line
                    curr_sample.extend([i[0]])  # start-times

                # And this is the main MIDI note line (triple stack)
                main_note = [i[1] + (i[2] * 16) + (i[3] * 16 * 128)]  # Main note == [duration / pitch / channel]

                if main_note != [0]:  # Main note error control...
                    curr_sample.extend(main_note)  # Main note == [duration / pitch / channel]

            train_list_y.append(train_data_y[song_number])
            train_list_x.append(curr_sample)

        song_number = song_number + 1

    # Potrebbe essere una criticit?? il fatto di fare il padding con 0
    # DA VERIFICARE
    # train_list_x = list_padding(train_list_x, 0)
    #
    # train_list_x = [np.array(x) for x in train_list_x]
    # train_list_x = np.array(train_list_x)
    #
    # train_list_y = np.array(train_list_y)



    #maxlen = SEQ_LEN *2

    #Puliamo la train list dalle liste con 0 valori non utili al training
    idx = [i for i, x in enumerate(train_list_x) if not x]
    for index in sorted(idx, reverse=True):
        del train_list_x[index]
        del train_list_y[index]

    #train_list_y = one_hot_encoding(train_list_y)

    train_list_x = np.array(train_list_x)
    train_list_y = np.array(train_list_y)

    train_list_x = keras.preprocessing.sequence.pad_sequences(train_list_x, padding="post")

    #train_list_x = normalize_input_data(train_list_x)

    return train_list_x, train_list_y


def normalize_input_data(list_x):
    max_values = len(np.unique(list_x))
    scaler = preprocessing.MinMaxScaler(feature_range=(0, max_values))
    train_list_x_norm = scaler.fit_transform(list_x)

    return train_list_x_norm


# Create a df with list of titles and a list of list containing the values for each title
def create_df(titles, values, filename):
    # Create a dataframe with the melody chords for each song processed and save that to a file
    df = pd.DataFrame(columns=['Song', 'Melody Chords'])
    df['Song'] = titles

    c = 0
    tot_c = 0
    for l in values:
        df.at[c, 'Melody Chords'] = l
        c = c + 1
        for el in l:
            tot_c = tot_c + 1

    #base_dir = r'C:\Users\andri\Desktop\Tesi\prove_classificazione\midi_classification copy\melody_chords_dataset'

    print("Total number of event wrote in db:")
    print(tot_c)

    #df.to_csv(os.path.join(base_dir, filename), index=False)
    df.to_csv(filename + '.csv', index=False)


def one_hot_encoding(a):
    b = np.zeros((len(a), max(a) + 1))
    b[np.arange(len(a)), a] = 1
    return b


def find_max_list(lists):
    list_len = [len(i) for i in lists]

    print("The max lenght of list is: ", max(list_len))

    return max(list_len)

#def train_tokenizer():