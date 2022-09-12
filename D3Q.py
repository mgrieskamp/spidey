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


class D3QAgent(torch.nn.Module):
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
        self.target = None  # Bellman equation target Q
        self.action = None  # Action taken
        self.q = torch.sum(torch.multiply(self.q_values, torch.nn.functional.one_hot(self.action, 4)), dim=1)
        self.loss = F.huber_loss(input=self.q, target=self.target, reduction='mean', delta=1.0)
        self.optimizer = optim.Adam(self.parameters(), weight_decay=0, lr=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

        # Weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    """
    Forward call through 4 convolution layers and final split linear layer for dueling.
    Input: Grayscale normalized image tensor
    Output: Tensor of q_values for each action
    """
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        conv_value, conv_adv = torch.split(x, 2)
        value = self.value_layer(conv_value)
        adv = self.adv_layer(conv_adv)
        adv_average = torch.mean(adv, dim=1, keepdim=True)
        q_values = value + adv - adv_average  # is this a 1x4 tensor???
        return q_values

    """
    Does a forward call and returns the index of the action with the highest Q value given a
    state. Meant to be used on the main CNN.
    Input: State sequence of (self.seq_size) frames
    Output: Index of best action in action list
    """
    def get_highest_q_action(self, state):
        with torch.no_grad():
            q_values = self.forward(state)
            best_action_index = torch.argmax(q_values, dim=1)  # are the dimensions correct???
        return best_action_index.item()

    """
    Does a forward call and returns the Q value of an action at a specified index given a state
    and an index. Meant to be used on the target CNN.
    Input: State sequence of (self.seq_size) frames, Index of specified action
    Output: Q value of specified action
    """
    def get_q_value_of_action(self, state, action_index):
        with torch.no_grad():
            q_values = self.forward(state)
            q_value = torch.flatten(q_values)[action_index]
        return q_value.item()


class Memory(object):
    def __init__(self, size=1000000, frame_h=200, frame_w=200, batch_size=32, seq_size=4):
        self.counter = 0
        self.current = 0
        self.size = size
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.batch_size = batch_size
        self.seq_size = seq_size

        self.frames = np.empty((self.size, self.frame_h, self.frame_w), dtype=np.uint8)
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.terminals = np.empty(self.size, dtype=np.bool)

        self.states = np.empty((self.batch_size, self.seq_size,
                                self.frame_h, self.frame_w), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.seq_size,
                                    self.frame_h, self.frame_w), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_memory(self, frame, action, reward, is_terminal):
        self.frames[self.current, ...] = frame
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminals[self.current] = is_terminal
        self.counter = max(self.counter, self.current + 1)
        self.current = (self.current + 1) % self.size

    def get_state(self, index):
        if self.counter == 0 or index < self.seq_size - 1:
            pass
        else:
            return self.frames[index - self.seq_size + 1: index + 1, ...]

    def get_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.seq_size, self.counter - 1)
                if index < self.seq_size:
                    continue
                if index >= self.current >= index - self.seq_size:
                    continue
                if self.terminals[index - self.seq_size:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        if self.counter < self.seq_size:
            pass
        self.get_indices()
        for i, idx in enumerate(self.indices):
            self.states[i] = self.get_state(idx - 1)
            self.new_states[i] = self.get_state(idx)
        # unknown dimensions???
        return self.states, self.actions, self.rewards, self.new_states, self.terminals