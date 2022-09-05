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
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

class deepQAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.reward = 0
        self.gamma = 0.9
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
        self.f1 = nn.Linear(11, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 3)
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
        """
        Numpy array of ? values:
            0) Player position
            1) Player velocity
            2) Player acceleration
            3) Platform 1 location
            4) Platform 2 location
            5) Platform 3 location
            6) Platform 4 location
            7) Platform 5 location
            8) Player is moving left
            9) Player is moving right
            10) Player is jumping
            11) Player is doing nothing

        """
        physics = spider.get_movement_coords()
        state = [physics[0], physics[1], physics[2]]
        pass

    def set_reward(self, spider):
        """
        Return the reward:
            -100 when game over.
            +10 when spider lands on platform
            -1 otherwise
        """
        self.reward = 0
        return self.reward

    def replay_memory(self):
        pass

    def store_transition(self):
        pass

    def execute_timestep(self):
        pass

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
            Sample random minibatch of transitions (s_t, a_t, r_t, s_t+1) from D
            Set y_j = r_j for terminal s_t+1 || Set y_j = r_j + gamma max_a' Q(s_t+1, a'; theta) for non-terminal s_t+1
            Perform gradient descent step on (y_j - Q(s_t, a_j; theta))^2
        end for
    end for
    """