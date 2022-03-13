import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.optim as optim
from torch.optim import lr_scheduler

import bombergym.settings as s
import pickle
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np

class CnnBoardNetwork(nn.Module):
    def __init__(self, output_size=len(s.ACTIONS)):
        super(CnnBoardNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.fc2(x.view(x.size(0), -1))
        return torch.sigmoid(x)

class GameplayDataset:
    def __init__(self, pckl_path):
        self.things = []
        with open(pckl_path, 'rb') as fd:
            self.data = pickle.load(fd)
        for episode in self.data:
            for data in episode:
                for step in data:
                    old_state, action, rew, obs = step
                    self.things.append((old_state, action))

    def __len__(self):
        return len(self.things)

    def __getitem__(self, item):
        state, action = self.things[item]
        action_oh = torch.zeros(len(s.ACTIONS), dtype=torch.float32)
        action_oh[action] = 1.
        state = torch.from_numpy(state.astype(np.float32))
        return state.swapaxes(0, 2), action_oh

gs = GameplayDataset('test.pckl')
print(len(gs))
def get_dataloaders():
    pass

def train_model(model, dataloaders, use_cuda, optimizer, scheduler, num_epochs,
                checkpoint_path_model, trained_epochs=0):
    best_loss = 1e10
    loss_fn = nn.BCELoss()

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

            epoch_samples = 0

            for dic in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
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
                epoch_samples += inputs.size(0)

            #print_metrics(metrics, epoch_samples, phase)
            #save_metrics(checkpoint_path_metrics, epoch, metrics, phase, epoch_samples)
            epoch_loss = loss / epoch_samples

            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'], "Loss", epoch_loss)

            # save the model weights
            if phase == 'val':
                if epoch_loss < best_loss:
                    print(f"saving best model to {checkpoint_path_model}")
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_path_model)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path_model))
    return model

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

    train_ds = GameplayDataset('test.pckl')
    train_size = int(len(train_ds) * .8)
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=32,
                                             shuffle=True, **kwargs)
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }
    model = CnnBoardNetwork()
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=0.00025)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=.95)

    model = train_model(model, dataloaders, use_cuda, optimizer_ft, exp_lr_scheduler,
                        num_epochs=10, checkpoint_path_model='model.pth')
