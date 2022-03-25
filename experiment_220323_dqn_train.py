import argparse
import collections
import random
import os

import gym
import numpy as np
import torch
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from bombergym.scenarios import classic, classic_with_opponents
from bombergym.environments import register

from experiment_220322_resnet_model import CnnboardResNet
from experiment_220322_cnnboard_val_metric import validate
from experiment_220323_cnnboard_common import get_transposer, detect_initial_configuration

def get_action(online_net, observation, transposer, epsilon=.05):
    if random.random() < .05:
        # Exploration: Choose random action
        return random.randrange(6)
    else:
        # Exploitation: Choose online network observationt action
        return transposer(online_net, observation)
    
def process_batch(online_net, target_net, buffer, device, batch_size=128, gamma=.99):
    if len(buffer) < batch_size: 
        return
    # Uniform sample from experience replay buffer - Lin, 1992
    sample = np.array(random.sample(buffer, batch_size), dtype=object)
    states = sample[:, 0]

    criterion = torch.nn.HuberLoss()

    state_batch = torch.from_numpy(np.concatenate(states)[np.newaxis, :].reshape(batch_size, 5, 17, 17)).to(device, torch.float32)
    next_state_batch = torch.from_numpy(np.concatenate(sample[:, 2])[np.newaxis, :].reshape(batch_size, 5, 17, 17)).to(device, torch.float32)
    actions = torch.from_numpy(sample[:, 1].astype(np.int64)).to(device)
    rewards = torch.from_numpy(sample[:, 3].astype(np.int32)).to(device)

    # Corresponding Q-values according to online network
    online_q_values = online_network(state_batch).gather(1, actions.unsqueeze(1)).squeeze()

    # Q-values of next observation according to target network (older)
    next_q_values = target_network(next_state_batch).max(dim=1).values

    # Bellmann Eqn., => Q-values we should have got according to actual rewards
    expected_q_values = next_q_values * gamma + rewards

    loss = criterion(online_q_values, expected_q_values)
    
    # Regular torch optimization routine - with gradient clipping
    optimizer.zero_grad()
    loss.backward()
    # Clamp Gradient: Recommended in "Human level control"
    # Nature Paper.
    for param in online_network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()
    
def train_model(
    env,
    online_network,
    target_network,
    optimizer,
    device,
    *,
    output_path=None,
    criterion=torch.nn.HuberLoss(),
    writer=None,
    epochs=100,
    epoch_size=256, # Define an epoch as n episodes
    batch_size=32,
    experience_replay_blen = 1000000, # Experience buffer replay length
    epoch_update_interval = 4 # How often to update target net, in epochs
):
    experience_replay_buffer = collections.deque(maxlen=experience_replay_blen)
    best_validation_winningfrac = 0.0

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        for episode in tqdm(range(epoch_size), total=epoch_size):
            obs = env.reset()
            initial_config = detect_initial_configuration(obs)
            transposer_raw = get_transposer(initial_config)
            transposer = lambda model, input: transposer_raw(model, input.astype(np.float32), device)
            while True:
                action = get_action(online_network, obs, transposer)
                next_obs, reward, done, other = env.step(action)
                if not done:
                    experience_replay_buffer.append((obs, action, next_obs, reward))
                obs = next_obs
                loss = process_batch(
                    online_network, 
                    target_network, 
                    experience_replay_buffer, 
                    device,
                    batch_size=batch_size
                )
                if loss is not None and writer is not None:
                    writer.add_scalar('DQN/loss', loss, (epoch*epoch_size)+episode)
                if done:
                    break

        # Validation phase - End of Epoch
        model_dict_cpucopy = {}
        for k, v in online_network.state_dict().items():
            model_dict_cpucopy[k] = v.cpu()

        winning_fraction = validate(model_dict_cpucopy)
        writer.add_scalar('Winning/val', winning_fraction, epoch)
        print(f'Val: Winning Fraction: {winning_fraction}')
        if winning_fraction > best_validation_winningfrac:
            best_validation_winningfrac = winning_fraction
            checkpoint_path_model = f'{output_path}/model{epoch}.pth'
            print(f"Saving best model to {checkpoint_path_model}")
            torch.save(online_network.state_dict(), checkpoint_path_model)
        if epoch % epoch_update_interval == 0:
            print("Updating target network")
            target_network.load_state_dict(online_network.state_dict())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--batch-size', help='batch size for optimization step', default=32, type=int)
    parser.add_argument('--output', help='Output Directory (Model/Tensorboard)', required=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    register()
    settings, agents = classic_with_opponents()

    env = gym.make('BomberGym-v4', args=settings, agents=agents)
    
    online_network = CnnboardResNet().to(device)
    target_network = CnnboardResNet().to(device)
    
    if args.resume is not None:
        online_network.load_state_dict(torch.load(args.resume))
        target_network.load_state_dict(torch.load(args.resume))
        print(f'Resuming from {args.resume}, sanity val...')
        winning_fraction = validate(torch.load(args.resume, map_location=torch.device('cpu')), n_jobs=2, n_episodes_per_job=10)
        print(f'Winning Frac of this model is {winning_fraction}')
    else:
        print('Warn: not resuming. Probably not what you want.')
        
    writer = SummaryWriter(args.output)
    optimizer = torch.optim.Adam(online_network.parameters(), lr=0.00025)
    train_model(
        env,
        online_network,
        target_network,
        optimizer,
        device,
        writer=writer,
        output_path=args.output,
        batch_size=args.batch_size,
        epoch_size=200,
    )