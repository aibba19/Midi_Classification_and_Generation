import os
import sys
import zipfile

from torch.utils.data import DataLoader

sys.path.append(
    r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\libraries\tegridy-tools\tegridy-tools')

base_dir = os.getcwd()

import TMIDIX
from GPT2RGAX import *

from Midi_Processing.preprocess_data import process_melody_chords

from collections import OrderedDict

import torch


class MusicSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand = random.randint(0, (self.data.size(0) - self.seq_len) // self.seq_len) * self.seq_len
        x = self.data[rand: rand + self.seq_len].long()
        trg = self.data[(rand + 1): (rand + 1) + self.seq_len].long()
        return x, trg

    def __len__(self):
        return self.data.size(0)

def load_model():
    full_path_to_model_checkpoint = \
        r"C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\Generation\Yoda_model\Yoda-Trained-Model\Yoda-Trained-Model.pth"  # @param {type:"string"}

    print('Loading the model...')
    config = GPTConfig(21938,
                           1024,
                           dim_feedforward=512,
                           n_layer=8,
                           n_head=8,
                           n_embd=512,
                           enable_rpr=True,
                           er_len=1024)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT(config)

    state_dict = torch.load(full_path_to_model_checkpoint, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module'
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    model.to(device)

    print(model.eval())

    return model

if __name__ == "__main__":
    model = load_model()
    train_data1 = process_melody_chords('generation')

    # @title Load processed INTs dataset

    SEQ_LEN = max_seq

    BATCH_SIZE = 16  # Change this to your specs

    # DO NOT FORGET TO ADJUST MODEL PARAMS IN GPT2RGAX module to your specs

    print('=' * 50)
    print('Loading training data...')

    data_train, data_val = torch.LongTensor(train_data1[:-(SEQ_LEN * BATCH_SIZE)]), torch.LongTensor(
        train_data1[-(SEQ_LEN * BATCH_SIZE) - 1:])

    train_dataset = MusicSamplerDataset(data_train, SEQ_LEN)
    val_dataset = MusicSamplerDataset(data_val, SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print('Total INTs in the dataset', len(train_data1))
    print('Total unique INTs in the dataset', len(set(train_data1)))
    print('Max INT in the dataset', max(train_data1))
    print('Min INT in the dataset', min(train_data1))
    print('=' * 50)

    print('Length of the dataset:', len(train_dataset))
    print('Length of data loader', len(train_loader))
    print('=' * 50)
    print('Done! Enjoy! :)')
    print('=' * 50)

    DIC_SIZE = max(train_data1) + 1

    init_step = 0
    lr = LR_DEFAULT_START
    lr_stepper = LrStepTracker(d_model, SCHEDULER_WARMUP_STEPS, init_step)
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=DIC_SIZE)
    train_loss_func = eval_loss_func

    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    lr_scheduler = LambdaLR(opt, lr_stepper.step)


    # Questi sono i valori che credo di dover cambiare con quelli estratti dal modello pre trainato se fa
    # best_eval_acc = 0.0
    best_eval_acc_epoch = -1
    # best_eval_loss = float("inf")
    best_eval_loss_epoch = -1

    best_eval_loss, best_eval_acc = eval_model(model, val_loader, eval_loss_func, num_iters=-1)

    best_acc_file = '/content/gpt2_rpr_acc.pth'
    best_loss_file = '/content/gpt2_rpr_loss.pth'
    loss_train, loss_val, acc_val = [], [], []

    epochs = 10

    for epoch in range(0, epochs):
        new_best = False

        loss = train(epoch + 1,
                     model, train_loader,
                     train_loss_func,
                     opt,
                     lr_scheduler,
                     num_iters=-1,
                     save_checkpoint_steps=4000)

        loss_train.append(loss)

        eval_loss, eval_acc = eval_model(model, val_loader, eval_loss_func, num_iters=-1)
        loss_val.append(eval_loss)
        acc_val.append(eval_acc)

        if (eval_acc > best_eval_acc):
            best_eval_acc = eval_acc
            best_eval_acc_epoch = epoch + 1
            torch.save(model.state_dict(), best_acc_file)
            new_best = True

        if (eval_loss < best_eval_loss):
            best_eval_loss = eval_loss
            best_eval_loss_epoch = epoch + 1
            torch.save(model.state_dict(), best_loss_file)
            new_best = True

        if (new_best):
            print("Best eval acc epoch:", best_eval_acc_epoch)
            print("Best eval acc:", best_eval_acc)
            print("")
            print("Best eval loss epoch:", best_eval_loss_epoch)
            print("Best eval loss:", best_eval_loss)
