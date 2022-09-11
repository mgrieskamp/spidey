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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DeepQAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.reward = 0
        self.gamma = 0.95
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.optimizer = None
        # Layers
        self.f1 = nn.Linear(10, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 5)
        # weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x

    def get_state(self, spider, plats):
        """ ADD MORE STATES - WHETHER PLATFORM HAS BEEN VISITED (plat.point)
        Numpy array of 10 values:
            1) Distance to center of closest plat
            2) Closest plat visited? (y/n = 1/0)
            3) Distance to center of next closest plat
            4) Next closest plat visited? (y/n = 1/0)
            ..) ... repeat until 5th closest
        """
        physics = spider.get_movement_coords()
        state = []
        plat_distances = []
        plats_and_dists = []
        for platform in plats:
            dist_to_plat = np.linalg.norm(platform.get_pos()[1] - physics[0])
            plat_distances.append(dist_to_plat)
            plat_and_dist = (platform, dist_to_plat)
            plats_and_dists.append(plat_and_dist)
        sorted_indices = np.argsort(plat_distances)
        for i in sorted_indices:
            state.append(int(plats_and_dists[i][0].point))
            state.append(plats_and_dists[i][1])
        return np.array(state[0:10])

    def set_reward(self, spider, game_over, old_state):
        """
        Return the reward:
            -100 when game over.
            +10 when spider lands on platform
            -0.1 otherwise
        """
        self.reward = 0
        # if game_over:
        #     self.reward -= 9
        #     return self.reward
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
        next_state_tensor = torch.from_numpy(next_state).cuda()
        state_tensor = torch.from_numpy(state).cuda()
        if terminal:
            target = reward
        else:
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor.float())[0]) #float?
        output_layer = self.forward(state_tensor.float()) #float?
        target_layer = output_layer.clone()
        target_layer[np.argmax(action)] = target  #0-dim tensor?
        target_layer.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output_layer, target_layer)
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
