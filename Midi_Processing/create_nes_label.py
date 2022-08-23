import pandas as pd
import json
import os
import sys

base_dir = os.getcwd()

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

label_lists = puzzle + fighting + puzzle + sport + shooting

# Function that scan the full nes db, and select the midi files that can be labelled and move these to the nes training folder
# and left the others in the original folder unlabelled for further tests
def create_nes_dataset():
    dataset_addr = r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\dataset\nes\nes_classification_test'
    n_labels = 5
    labels = []
    #Dalla cartella con tutti i file originali assegno la label a tutti quelli possibili
    for (dirpath, dirnames, filenames) in os.walk(dataset_addr):

        for file in filenames:
            l = assign_label(file,n_labels)
            if not all(v == 0 for v in l):
                labels.append(l)
                os.replace(os.path.join(dirpath, file),
                           os.path.join(r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\dataset\nes_training', file))
        #filez += [os.path.join(dirpath, file) for file in filenames]

        # Creo array con i titoli delle canzoni

        #labels = assign_label(file)
    dataset_training = r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\dataset\nes_training'

    #Check Db structure

    #Count How many files are left without labels
    #num_unlabelled = len([x for x in labels if x is None])
    num_unlabelled = 0

    #Array containing the count for how many songs with each label are present
    #in this order : rpg,sport,fighting,shooting,puzzle
    labels_count = [0] * n_labels

    for x in labels:
        # if all(v == 0 for v in x):
        #     num_unlabelled = num_unlabelled + 1
        #     continue
        for v in range(len(x)):
            if x[v] == 1:
                labels_count[v] = labels_count[v] + 1


    print("Labels count in this order, rpg,sport,fighting,shooting,puzzle : " ,labels_count)



    df = pd.DataFrame(columns=['title', 'label'])

    for (dirpath, dirnames, filenames1) in os.walk(dataset_training):
        f = filenames1

    df['title'] = f
    df['label'] = labels

    #Take for training only the songs labelled
    df = df[~df['label'].isnull()]

    df.to_csv(r"../Classification/nes_labelled.csv", sep=';', index=False)


#def check_nes_db_composition():



# Check if a file is one of those of the selected games
def is_present(file_name, search_list=label_lists):
    present = False
    for name in search_list:
        if name in file_name:
            present = True
    return present

# Function used to assign a label with respect to the file name
# def assign_label(file_name):
#
#     if is_present(file_name, rpg):
#         return 0
#     elif is_present(file_name, sport):
#         return 1
#     elif is_present(file_name, fighting):
#         return 2
#     elif is_present(file_name, shooting):
#         return 3
#     elif is_present(file_name, puzzle):
#         return 4

def assign_label(file_name,n_labels):

    labels = [0] * n_labels

    if is_present(file_name, rpg):
        labels[0] = 1
    if is_present(file_name, sport):
        labels[1] = 1
    if is_present(file_name, fighting):
        labels[2] = 1
    if is_present(file_name, shooting):
        labels[3] = 1
    if is_present(file_name, puzzle):
        labels[4] = 1

    return labels
