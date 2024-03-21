import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding='same')

    def forward(self, x):
        x = x.to(device)
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)

class DeepQNetwork(nn.Module):
    def __init__(self, n_observations = 256, n_actions = 4, batch_size = 10):
        super(DeepQNetwork, self).__init__()
        # setting up some hyperparameters
        self.lr = 0.001
        self.gamma = 0.99
        self.start_epsilon = 0.99
        self.epsilon = 0.99
        self.epsilon_dec = 0.99
        self.end_epsilon = 0.1
        self.batch_size = batch_size
        self.steps_done = 0
        # self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        self.n_episodes = 0
        self.n_iterations = 0

        # setting up the network 
        # self.Layer1 = nn.Sequential(nn.Linear(n_observations, 64), nn.ReLU(inplace=True))
        # self.Layer2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        # self.FinalLayer = nn.Sequential(nn.Linear(64, n_actions))
        # self._create_weights()
        self.conv1 = ConvBlock(16, 2048)
        self.conv2 = ConvBlock(2048, 2048)
        self.conv3 = ConvBlock(2048, 2048)
        self.dense1 = nn.Linear(2048 * 16, 1024)
        self.dense6 = nn.Linear(1024, 4)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()


    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    # def forward(self, x):
    #     x = self.Layer1(x)
    #     x = self.Layer2(x)
    #     x = self.FinalLayer(x)

    #     return x
        
    def forward(self, x):
        x = x.to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = self.dropout(self.dense1(x))
        return self.dense6(x)
    
    def execute_action(self, current_state):
        with torch.no_grad():
            self.steps_done += 1
            if np.random.uniform(0,1) < self.epsilon:
                action = np.random.choice(4,1)[0]
                return action
            #print(f"size of curr: {current_state.size()}" )
            q_values = self.forward(current_state)
            q_cpu = q_values.cpu()
            action = np.argmax(q_cpu.detach().numpy())
            # return the action, not a list 
            # print (f"action: {action}")
            return action
            # return 

    def update_epsilon(self):
        # decrease the epsilon value by a factor of 0.005
        # print (f"epsilon: {self.epsilon}")
        self.epsilon = max (self.end_epsilon, self.start_epsilon * (self.epsilon_dec ** self.steps_done))
        # print (f"epsilon: {self.epsilon}")


