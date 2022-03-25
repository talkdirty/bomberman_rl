import numpy as np
import re
import argparse
import lzma
import pickle
import time
import random
from collections import defaultdict
from tqdm import tqdm
import os
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import bombergym.settings as s
from experiment_220324_deepmind_val import validate
from experiment_220324_deepmind_arch import DeepmindAtariCNN, DeepmindAtariCNNDeep

class GameplayDataset():
    def __init__(self, directory):
        self.directory = directory
        self.files = os.listdir(directory)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        try:
            with lzma.open(f'{self.directory}/{self.files[item]}') as fd:
                state, action = pickle.load(fd)[:2]
        except EOFError:
            print(f'Warn: {item}, {self.files[item]} is broken.')
            return random.choice(self)
        action_oh = torch.zeros(len(s.ACTIONS), dtype=torch.float32)
        action_oh[action] = 1.
        state = torch.from_numpy(state.astype(np.float32))
        return state, action_oh

def train_model(
        model, 
        criterion, 
        dataloaders, 
        device, 
        optimizer, 
        scheduler, 
        batch_size, 
        writer,
        output_path,
        num_epochs=25, 
        start_epoch=0,
        is_deep=False,
        ):
    since = time.time()

    best_validation_winningfrac = 0.0

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            model.train()

            running_loss = 0.0
            running_corrects = 0
            
            dataset_size = len(dataloaders[phase]) * batch_size

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.argmax(axis=1))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        # Validation phase
        model_dict_cpucopy = {}
        for k, v in model.state_dict().items():
            model_dict_cpucopy[k] = v.cpu()

        winning_fraction = validate(model_dict_cpucopy, n_episodes=100)
        writer.add_scalar('Winning/val', winning_fraction, epoch)
        print(f'Winning fraction is {winning_fraction}')
        if winning_fraction > best_validation_winningfrac:
            best_validation_winningfrac = winning_fraction
            checkpoint_path_model = f'{output_path}/model{epoch}-deep{is_deep}.pth'
            print(f"Saving best model to {checkpoint_path_model}")
            torch.save(model.state_dict(), checkpoint_path_model)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='Output for tensorboard & Checkpoints', required=True)
    parser.add_argument('--resume', help='Resume checkpoint', required=False)
    parser.add_argument('--data', help='Training data Input', required=True)
    parser.add_argument('--epochs', default=60, required=False, type=int)
    parser.add_argument('--deep', action='store_true', required=False, default=False)

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    #train_ds = GameplayDataset('data_frames_randomness/')
    train_ds = GameplayDataset(args.data)
    print(f'Loaded Data (len={len(train_ds)})')
    train_size = int(len(train_ds) * .9)
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    #val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, **kwargs)
    dataloaders = {
        "train": train_loader,
        #"val": val_loader,
    }
    if not args.deep:
        model = DeepmindAtariCNN().to(device)
    else:
        model = DeepmindAtariCNNDeep().to(device)


    optimizer_ft = optim.Adam(model.parameters(), lr=0.0025)
    #optimizer_ft = optim.SGD(model.parameters(), lr=0.01)
    #optimizer_ft = optim.RMSprop(model.parameters(), lr=0.00025, momentum=0.95)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=.1)
    writer = SummaryWriter(args.output)

    last_epoch = -1
    if args.resume is not None:
        epoch = re.findall(r'\d+', args.resume.split('/')[-1])
        if len(epoch):
            epoch = int(epoch[0])
            last_epoch = epoch
            model.load_state_dict(torch.load(args.resume))
            for _ in range(last_epoch):
                exp_lr_scheduler.step()
            print(f'Resuming training from epoch {epoch}')
        else:
            raise RuntimeError('Could not parse epoch, Not resuming training.')

    criterion = nn.CrossEntropyLoss()
    model = train_model(
            model, 
            criterion, 
            dataloaders, 
            device, 
            optimizer_ft, 
            exp_lr_scheduler, 
            batch_size,
            writer, 
            args.output, 
            num_epochs=args.epochs,
            start_epoch=last_epoch + 1,
            is_deep=args.deep,
    )