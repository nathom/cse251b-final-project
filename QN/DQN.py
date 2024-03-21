import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, n_observations = 256, n_actions = 4, batch_size = 10):
        super(DeepQNetwork, self).__init__()
        # setting up some hyperparameters
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon = 0.99
        self.epsilon_dec = 0.005
        self.batch_size = batch_size
        # self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        self.n_episodes = 0
        self.n_iterations = 0

        # setting up the network 
        self.Layer1 = nn.Sequential(nn.Linear(n_observations, 64), nn.ReLU(inplace=True))
        self.Layer2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.FinalLayer = nn.Sequential(nn.Linear(64, n_actions))
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.FinalLayer(x)

        return x
    
    def execute_action(self, current_state):
        with torch.no_grad():
            if np.random.uniform(0,1) < self.epsilon:
                return np.random.choice(4,1)
            #print(f"size of curr: {current_state.size()}" )
            q_values = self.forward(current_state)
            return np.argmax(q_values)

    def update_epsilon(self):
        self.epsilon = self.epsilon * np.exp(-self.epsilon_dec)   
    


