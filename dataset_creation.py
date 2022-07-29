import pandas as pd
import json
import os
import sys

base_dir = os.getcwd()

with open(os.path.join(base_dir, "lists/adventure.json"), "r") as fp:
    adventure = json.load(fp)
with open(os.path.join(base_dir, "lists/fighting.json"), "r") as fp:
    fighting = json.load(fp)
with open(os.path.join(base_dir, "lists/puzzle.json"), "r") as fp:
    puzzle = json.load(fp)
with open(os.path.join(base_dir, "lists/sport.json"), "r") as fp:
    sport = json.load(fp)
with open(os.path.join(base_dir, "lists/shooting.json"), "r") as fp:
    shooting = json.load(fp)

label_lists = adventure + fighting + puzzle + sport + shooting

def create_nes_dataset():
    dataset_addr = r'C:\Users\andri\Desktop\Tesi\prove_classificazione\midi_classification copy\dataset\nes\nes'

    labels = []
    #Dalla cartella con tutti i file originali assegno la label a tutti quelli possibili
    for (dirpath, dirnames, filenames) in os.walk(dataset_addr):

        for file in filenames:
            l = assign_label(file)
            if l:
                labels.append(l)
                os.replace(os.path.join(dirpath, file),
                           os.path.join(r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\dataset\nes_training', file))
        #filez += [os.path.join(dirpath, file) for file in filenames]

        # Creo array con i titoli delle canzoni

        #labels = assign_label(file)
    dataset_training = r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\dataset\nes_training'

    #Count How many files are left without labels
    num_unlabelled = len([x for x in labels if x is None])
    print("Songs left unlabelled: ", num_unlabelled)


    df = pd.DataFrame(columns=['title', 'label'])

    for (dirpath, dirnames, filenames1) in os.walk(dataset_training):
        f = filenames1

    df['title'] = f
    df['label'] = labels

    #Take for training only the songs labelled
    df = df[~df['label'].isnull()]

    df.to_csv('nes_labelled.csv', sep=';', index=False)



# Check if a file is one of those of the selected games
def is_present(file_name, search_list=label_lists):
    present = False
    for name in search_list:
        if name in file_name:
            present = True
    return present

# Function used to assign a label with respect to the file name
def assign_label(file_name):

    if is_present(file_name, adventure):
        return 1
    elif is_present(file_name, sport):
        return 2
    elif is_present(file_name, fighting):
        return 3
    elif is_present(file_name, shooting):
        return 4
    elif is_present(file_name, puzzle):
        return 5
