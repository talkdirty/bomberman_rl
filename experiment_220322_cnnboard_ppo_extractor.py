import gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env

from testresnet import resnet18
from bombergym.scenarios import classic, classic_with_opponents, coin_heaven
from bombergym.environments import register

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, weights_path="dqn_test2.pth"):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.backbone = resnet18(
            norm_layer=lambda channels: torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-5, affine=True),
            num_classes=6
        )
        print(f"Loading weights from {weights_path}")
        self.backbone.load_state_dict(torch.load(weights_path))
        self.linear = nn.Sequential(nn.Linear(6, features_dim), nn.ReLU())

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return self.linear(x)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

register()
settings, agents = classic_with_opponents()
# env = gym.make('BomberGym-v4', args=settings, agents=agents)
env = make_vec_env("BomberGym-v4", n_envs=4, env_kwargs={'args': settings, 'agents': agents})
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(500000)
