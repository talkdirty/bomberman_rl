import torch
import torch.nn.functional as F

class DeepmindAtariCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 16, kernel_size=(4, 4), stride=4)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(2, 2), stride=2)
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 6)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X
    
class DeepmindAtariCNNDeep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 32, kernel_size=(3, 3), stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(2, 2), stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(2, 2), stride=2)
        self.fc1 = torch.nn.Linear(128, 256) # tmp for doublespace
        self.fc2 = torch.nn.Linear(256, 6)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X