import os

from create_nes_label import *
from nes_videogame_categorization import *
from preprocess_data import *


if __name__ == "__main__":

    print('Creating dataset dirs...')
    path_nes_training = r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\dataset\nes\nes_training'
    dir = os.listdir(path_nes_training)

    if len(dir) == 0:
        create_nes_dataset()
    else:
        print("Directory with nes songs labelled already exists")
    #if not os.path.exists(os.path.join(base_dir, r'dataset\nes_training')):
    #    create_nes_dataset()



    print('Nes Dataset folder for classification created!')

    #os.chdir(r'C:\Users\andri\Desktop\Tesi\prove_classificazione\midi_classification copy\tegridy-tools\tegridy-tools')
    #os.chdir(base_dir)

    #print("We are inside this folder" + str(os.path.abspath(os.getcwd())))
    #print('Loading TMIDIX module...')

