import lzma
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import os
import bombergym.settings as s
import pickle
import time
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from testresnet import resnet18

class GameplayDataset():
    def __init__(self, directory):
        self.directory = directory
        self.files = os.listdir(directory)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        try:
            with lzma.open(f'{self.directory}/{self.files[item]}') as fd:
                state, action, _ = pickle.load(fd)
        except EOFError:
            print(f'Warn: {item}, {self.files[item]} is broken.')
            return random.choice(self)
        action_oh = torch.zeros(len(s.ACTIONS), dtype=torch.float32)
        action_oh[action] = 1.
        state = torch.from_numpy(state.astype(np.float32))
        return state, action_oh

###copypasta start
import time
import copy 
def train_model(model, criterion, dataloaders, device, optimizer, scheduler, batch_size, num_epochs=25, checkpoint_path_model='model.pth'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.argmax(axis=1))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"saving best model to {checkpoint_path_model}")
                torch.save(model.state_dict(), checkpoint_path_model)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
###copypasta end

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    #train_ds = GameplayDataset('data_frames_randomness/')
    train_ds = GameplayDataset('data_augmented_doublespace/')
    print(f'Loaded Data (len={len(train_ds)})')
    train_size = int(len(train_ds) * .9)
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, **kwargs)
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }
    model = resnet18(
        norm_layer=lambda channels: torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-5, affine=True),
        num_classes=6
    ).to(device)
    #model = CnnBoardNetwork()
    if use_cuda:
        model = model.cuda()
    optimizer_ft = optim.Adam(model.parameters(), lr=0.0025)
    #optimizer_ft = optim.SGD(model.parameters(), lr=0.01)
    #optimizer_ft = optim.RMSprop(model.parameters(), lr=0.00025, momentum=0.95)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=.5)
    writer = SummaryWriter('testlog')

    criterion = nn.CrossEntropyLoss()
    model = train_model(model, criterion, dataloaders, device, optimizer_ft, exp_lr_scheduler, batch_size,
            num_epochs=300, checkpoint_path_model='model.pth')
