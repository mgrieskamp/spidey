import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import params

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DeepQAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.reward = 0
        self.gamma = 0.95
        self.learning_rate = params['learning_rate']
        self.epsilon = 1
        # self.first_layer = params['first_layer_size']
        # self.second_layer = params['second_layer_size']
        # self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']

        self.input = None

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding='valid', bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding='valid', bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='valid', bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=1024, kernel_size=7, stride=4, padding='valid', bias=False)

        # Output Layers
        self.value_layer = nn.Linear(512, 1)
        self.adv_layer = nn.Linear(512, 4)

        # # Advantage and Value Functions
        # self.value_stream, self.advantage_stream = torch.split(self.forward(self.input), 512, dim=3)
        # self.value_stream = torch.flatten(self.value_stream)
        # self.advantage_stream = torch.flatten(self.advantage_stream)
        # self.value = nn.Linear(512, 1).forward(self.value_stream)
        # self.advantage = nn.Linear(512, 4).forward(self.advantage_stream)     # num actions

        self.q_values = None
        self.best_action = None

        # Updates
        # self.target = None  # bellman equation target
        # self.action = None  # Action taken
        # self.q = torch.sum(torch.multiply(self.q_values, torch.nn.functional.one_hot(self.action, 4)), dim=1)
        # self.loss = F.huber_loss(input=self.q, target=self.target, reduction='mean', delta=1.0)
        self.optimizer = optim.Adam(self.parameters(), weight_decay=0, lr=self.learning_rate)
        # self.update = self.optimizer.minimize(self.loss)

        # weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        conv_value, conv_adv = torch.split(x, 2)
        value = self.value_layer(conv_value)
        adv = self.adv_layer(conv_adv)
        adv_average = torch.mean(adv, dim=1, keepdim=True)
        self.q_values = value + adv - adv_average
        self.best_action = torch.argmax(self.q_values, dim=1)

        return x

    # def get_state(self, spider, plats):
    #     """ ADD MORE STATES - WHETHER PLATFORM HAS BEEN VISITED (plat.point)
    #     Numpy array of 10 values:
    #         1) Distance to center of closest plat
    #         2) Closest plat visited? (y/n = 1/0)
    #         3) Distance to center of next closest plat
    #         4) Next closest plat visited? (y/n = 1/0)
    #         ..) ... repeat until 5th closest
    #     """
    #
    #     grid = np.zeros(shape=(30, 20))
    #     width = 20
    #     height = 15
    #     for plat in plats:
    #         left = plat.rect.left
    #         right = plat.rect.right
    #         top = plat.rect.top
    #         bottom = plat.rect.bottom
    #         l_bound = int(np.floor(left / width))
    #         r_bound = int(np.floor(right / width))
    #         t_bound = int(np.floor(top / height))
    #         b_bound = int(np.floor(bottom / height))
    #         for i in range(l_bound, r_bound + 1):
    #             for j in range(t_bound, b_bound + 1):
    #                 if not i >= 20 and not j >= 30:
    #                     if plat.point:
    #                         grid[j][i] = 1
    #                     else:
    #                         grid[j][i] = -1
    #     l_bound = int(np.floor(spider.rect.left / width))
    #     r_bound = int(np.ceil(spider.rect.right / width))
    #     t_bound = int(np.floor(spider.rect.top / height))
    #     b_bound = int(np.floor(spider.rect.bottom / height))
    #     print(spider.rect.top)
    #     for i in range(l_bound, r_bound + 1):
    #         for j in range(t_bound, b_bound + 1):
    #             if not i >= 20 and not j >= 30:
    #                 print("(" + str(i) + ", " + str(j) + ")")
    #                 grid[j][i] = 2
    #     return np.reshape(grid, 600)

        # physics = spider.get_movement_coords()
        # state = []
        # plat_distances = []
        # plats_and_dists = []
        # for platform in plats:
        #     dist_to_plat = np.linalg.norm(platform.get_pos()[1] - physics[0])
        #     plat_distances.append(dist_to_plat)
        #     plat_and_dist = (platform, dist_to_plat)
        #     plats_and_dists.append(plat_and_dist)
        # sorted_indices = np.argsort(plat_distances)
        # for i in sorted_indices:
        #     state.append(int(plats_and_dists[i][0].point))
        #     state.append(plats_and_dists[i][1])
        # return np.array(state[0:10])

    def set_reward(self, spider, game_over, old_state):
        """
        Return the reward:
            -100 when game over.
            +10 when spider lands on platform
            -0.1 otherwise
        """
        self.reward = 0
        if game_over:
            self.reward -= 5
            return self.reward
        if spider.on_platform:
            self.reward += 0
        if spider.pos.y < old_state[1]:
            self.reward += 0
        if spider.pos.y > old_state[1]:
            self.reward -= 0
        if spider.new_landing:
            self.reward += 10
        return self.reward

    def train_short_term(self, state, action, reward, next_state, terminal):
        self.train()
        torch.set_grad_enabled(True)
        if terminal:
            target = reward
        else:
            target = reward + self.gamma * torch.max(self.forward(next_state.float())[0]) #float?  #TODO: this needs changing
        output = torch.sum(torch.multiply(self.q_values, torch.nn.functional.one_hot(self.action, 4)), dim=1)
        loss = F.huber_loss(input=output, target=target, reduction='mean', delta=1.0)

        # output_layer = self.forward(state.float()) #float?
        # target_layer = output_layer.clone()
        # target_layer[np.argmax(action)] = self.target  #0-dim tensor?
        # target_layer.detach()
        # self.optimizer.zero_grad()
        # loss = F.mse_loss(output_layer, target_layer)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def replay_memory(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for (state, action, reward, next_state, terminal) in minibatch:
            self.train_short_term(state, action, reward, next_state, terminal)

    def store_transition(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))


    """
    (Pseudocode)
    Deep Q-learning with experience replay:
    
    Initialize replay memory D to capacity N
    Initialize action-value function Q with random weights
    for episode 1, M do
        Initialize state s_t
        for t = 1, T do
            With probability e select a random action a_t
            otherwise select a_t = max_a Q*(s_t, a; theta)
            Execute action a_t and observe reward r_t and state s_t+1
            Store transition (s_t, a_t, r_t, s_t+1) in D
            Set s_t+1 = s_t
            Sample random minibatch of transitions (s_t, a_t, r_t, s_t+1) from D (deepQAgent.replay_new(D, batch_size)
        end for
    end for
    
    
    replay_new(memory, batch_size):
        For every transition j in minibatch:
            if s_j+1 terminal:
                y_j = r_j
            else:
                # get target value; i.e. max possible predicted future reward by choosing best possible next action
                y_j = r_j + gamma max_a' Q(s_t+1, a'; theta)
            # Perform gradient descent step on (y_j - Q(s_t, a_j; theta))^2 
            Calculate loss L = 1/N sum((Q(s_j, i_j) - y_j)^2)
            Update Q using SGD by minimizing loss L
    """
