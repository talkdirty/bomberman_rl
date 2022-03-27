import re
import copy
import random
import argparse
import collections
import time

import gym
from bombergym.scenarios import classic_with_opponents
from bombergym.environments import register

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import bombergym.settings as s
from experiment_220322_cnnboard_val_metric import validate
from experiment_220322_resnet_model import CnnboardResNet
from experiment_220326_cnnboard_v5_common import get_transposer, detect_initial_configuration

class PrioritizedReplayDataset():
    def __init__(self, experiences):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, ix):
        state, action = self.experiences[ix][:2]
        action_oh = torch.zeros(len(s.ACTIONS), dtype=torch.float32)
        action_oh[action] = 1.
        state = torch.from_numpy(state.astype(np.float32))
        return state, action_oh

def gather_experience(model, env, device):
    experience_buffer = []
    selective_buffer = []
    epsilon = .05
    sliding_window_size = 10
    goodness_thresh = 0 # todo adjust
    reward_trigger_thresh = 0 # todo maybe?
    selective_buffer_size = 50000

    while True:
        obs = env.reset()
        initial_config = detect_initial_configuration(obs)
        transposer = get_transposer(initial_config)
        while True:
            previous_obs = obs
            if random.random() > epsilon:
                action = transposer(model, obs, device)
                obs, rew, done, other = env.step(action)
            else:
                action = random.randrange(6)
                obs, rew, done, other = env.step(action)
            experience_buffer.append((previous_obs, action, obs, rew, done, other))
            if done:
                # Process the experience for a trigger event
                experience_start = 0
                while True:
                    experience_end = experience_start + sliding_window_size
                    if experience_end >= len(experience_buffer):
                        break # we're done here, it was already processed
                    experience_slice = experience_buffer[experience_start:experience_end]
                    rewards = np.array([a[3] for a in experience_slice])
                    if rewards[-1] == np.max(rewards) and np.mean(rewards) > goodness_thresh and rewards[-1] > reward_trigger_thresh:
                        # Be VERY specific about which experiecnes we want to enforce
                        # WARN if we kill ourselves with the bomb this is currently also in
                        if 'COIN_COLLECTED' in experience_buffer[experience_end-1][5]['events'] or 'KILLED_OPPONENT' in experience_buffer[experience_end-1][5]['events']:
                            # print(f'found a good experience: ', rewards, experience_buffer[experience_end-1][4]['events'])
                            selective_buffer += experience_slice
                            experience_start = experience_end # jump forwards, avoids need for NMS
                        else:
                            experience_start += 1
                    else:
                        experience_start += 1
                break
        if len(selective_buffer) > selective_buffer_size:
            print('Gathered enough.')
            return selective_buffer

def train_model(
        env,
        model, 
        criterion, 
        device, 
        optimizer, 
        batch_size, 
        writer,
        output_path,
        baseline=0,
        scheduler=None, 
        num_epochs=25, 
        start_epoch=0,
        ):
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_validation_winningfrac = baseline # smth like this

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            model.train()

            running_loss = 0.0
            running_corrects = 0
            
            print('Gathering Experiences..')
            ds = PrioritizedReplayDataset(gather_experience(model, env, device))
            dl = DataLoader(ds, batch_size = batch_size)

            # Iterate over data.
            for inputs, labels in dl:
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
                if scheduler is not None:
                    scheduler.step()

        # Validation phase
        model_dict_cpucopy = {}
        for k, v in model.state_dict().items():
            model_dict_cpucopy[k] = v.cpu()

        winning_fraction = validate(model_dict_cpucopy, n=1000)
        writer.add_scalar('Winning/val', winning_fraction, epoch)
        if winning_fraction > best_validation_winningfrac:
            best_validation_winningfrac = winning_fraction
            checkpoint_path_model = f'{output_path}/model{epoch}.pth'
            print(f"Saving best model to {checkpoint_path_model}")
            torch.save(model.state_dict(), checkpoint_path_model)
            best_model = copy.deepcopy(model.state_dict())
        else:
            print(f'Did not exceed best model performance. Resetting.')
            model.load_state_dict(best_model)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='Output for tensorboard & Checkpoints', required=True)
    parser.add_argument('--resume', help='Resume checkpoint path', default='out_220326_supervised_resnet18_v5/model18.pth')
    parser.add_argument('--epochs', default=300, required=False)

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    model = CnnboardResNet().to(device)
    model.load_state_dict(torch.load(args.resume, map_location=device))
    register()
    settings, agents = classic_with_opponents()

    env = gym.make('BomberGym-v5', args=settings, agents=agents)

    #optimizer_ft = optim.Adam(model.parameters(), lr=0.0025)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.00005)
    #optimizer_ft = optim.RMSprop(model.parameters(), lr=0.00025, momentum=0.95)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=.1)
    writer = SummaryWriter(args.output)

    last_epoch = -1
    if args.resume is not None:
        epoch = re.findall(r'\d+', args.resume.split('/')[-1])
        if len(epoch):
            epoch = int(epoch[0])
            last_epoch = epoch
            model.load_state_dict(torch.load(args.resume))
            #for _ in range(last_epoch):
            #    exp_lr_scheduler.step()
            print(f'Resuming training from epoch {epoch}')
        else:
            raise RuntimeError('Could not parse epoch, Not resuming training.')

    model_dict_cpu = torch.load(args.resume, map_location=torch.device('cpu'))
    print('Computing baseline winning fraction')
    winning_fraction = validate(model_dict_cpu, n=1000)
    print(f'Baseline is {winning_fraction}')
    criterion = nn.CrossEntropyLoss()
    model = train_model(
            env,
            model, 
            criterion, 
            device, 
            optimizer_ft, 
            batch_size,
            writer, 
            args.output, 
            baseline=winning_fraction,
            #scheduler=exp_lr_scheduler, 
            num_epochs=args.epochs,
            start_epoch=last_epoch + 1
    )
