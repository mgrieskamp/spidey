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
        self.memory = None
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

        # Updates
        self.target = None  # bellman equation target
        self.action = None  # Action taken
        self.q = torch.sum(torch.multiply(self.q_values, torch.nn.functional.one_hot(self.action, 4)), dim=1)
        self.loss = F.huber_loss(input=self.q, target=self.target, reduction='mean', delta=1.0)
        self.optimizer = optim.Adam(self.parameters(), weight_decay=0, lr=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

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
        q_values = value + adv - adv_average # is this a 1x4 tensor?
        return q_values

    def select_action(self):
        pass
