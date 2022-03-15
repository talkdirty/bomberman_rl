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

class CnnBoardNetwork(nn.Module):
    def __init__(self, output_size=len(s.ACTIONS)):
        super(CnnBoardNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.fc2(x)
        return x

class GameplayDataset():
    def __init__(self, directory):
        self.directory = directory
        self.files = os.listdir(directory)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        try:
            with lzma.open(f'{self.directory}/{self.files[item]}') as fd:
                state, action, _, _ = pickle.load(fd)
        except EOFError:
            print(f'Warn: {item}, {self.files[item]} is broken.')
            return random.choice(self)
        action_oh = torch.zeros(len(s.ACTIONS), dtype=torch.float32)
        action_oh[action] = 1.
        state = torch.from_numpy(state.astype(np.float32))
        return state.swapaxes(0, 2), action_oh

def train_model(model, dataloaders, use_cuda, optimizer, scheduler, num_epochs,
                checkpoint_path_model, trained_epochs=0, writer=None):
    best_loss = 1e10
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(trained_epochs, num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for idx, dic in enumerate(tqdm(dataloaders[phase], total=len(dataloaders[phase]))):
                inputs, labels = dic
                if use_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloaders[phase])
            if phase == 'train':
                scheduler.step()
                writer.add_scalar('Loss/train', epoch_loss, epoch * len(dataloaders[phase]))
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'], "Loss", epoch_loss, loss)

            # save the model weights
            if phase == 'val':
                writer.add_scalar('Loss/val', epoch_loss, epoch * len(dataloaders[phase]))
                if epoch_loss < best_loss:
                    print(f"saving best model to {checkpoint_path_model}")
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_path_model)

            running_loss = 0.0
            writer.flush()

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path_model))
    return model

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_ds = GameplayDataset('data_frames_randomness/')
    print(f'Loaded Data (len={len(train_ds)})')
    train_size = int(len(train_ds) * .9)
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=True, **kwargs)
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }
    model = CnnBoardNetwork()
    if use_cuda:
        model = model.cuda()
    optimizer_ft = optim.Adam(model.parameters(), lr=0.0025)
    #optimizer_ft = optim.SGD(model.parameters(), lr=0.01)
    #optimizer_ft = optim.RMSprop(model.parameters(), lr=0.00025, momentum=0.95)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=.1)
    writer = SummaryWriter('testlog')

    model = train_model(model, dataloaders, use_cuda, optimizer_ft, exp_lr_scheduler,
            num_epochs=300, checkpoint_path_model='model.pth', writer=writer)
