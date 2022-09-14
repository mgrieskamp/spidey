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

        # Convolutional Layers - (N, C_in, H, W) -> (N, C_out, H_out, W_out)
        # Dimension will be: (32, 4, 200, 200) -> (32, C_out, H_out, W_out)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding='valid', bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding='valid', bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='valid', bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=1024, kernel_size=8, stride=4, padding='valid', bias=False)

        # Output Layers - (*, C_in) -> (*, C_out)
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
    Input: Grayscale normalized image tensor, mini-batches allowed. Dim: (N, C_in, H, W)
    Output: (N, C_out, H, W) Tensor of q_values for each action
    """
    def forward(self, x):
        x = F.relu(self.conv1(x))  # (32, 4, 200, 200) tensor in
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # (32, 1024, 1, 1) tensor out
        conv_value, conv_adv = torch.split(x, 2, dim=1)  # we need to split this in the right dimension. (C_out)
        conv_value = torch.permute(conv_value, (0, 2, 3, 1))  # (32, 1, 1, 512) tensor
        conv_adv = torch.permute(conv_adv, (0, 2, 3, 1))  # (32, 1, 1, 512) tensor
        value = self.value_layer(conv_value)  # (32, 1, 1, 512) -> (32, 1, 1, 1)
        adv = self.adv_layer(conv_adv)  # (32, 1, 1, 512) -> (32, 1, 1, 4)
        adv_average = torch.mean(adv, dim=3, keepdim=True)  # (32, 1, 1, 1)
        q_values = torch.subtract(torch.add(adv, value), adv_average)  # broadcast (32, 1, 1, 4)
        return torch.flatten(q_values, start_dim=1, end_dim=2)  # (32, 4)

    def get_reward(self, spider, game_over):
        """
        Return the reward:
            -100 when game over.
            +10 when spider lands on platform
            -0.1 otherwise
        """
        self.reward = 0
        if game_over:
            self.reward -= 0
            return self.reward
        if spider.new_landing:
            self.reward += 1
        return self.reward

    """
    Does a forward call and returns the index of the action with the highest Q value given a
    state. Meant to be used on the main CNN.
    Input: State sequence of (self.seq_size) frames
    Output: Index of best action in action list (int)
    """
    def get_highest_q_action(self, state):
        with torch.no_grad():
            q_values = self.forward(state)  # (N, 4, 200, 200) -> (N, 1, 1, 4)
            if q_values.dim() == 3:  # if single state (1, 1, 4)
                q_values = torch.flatten(q_values)  # (1, 1, 4) -> (4)
                best_action_index = torch.argmax(q_values)  # (1)
                return best_action_index.item()  # int
            else:  # if batch of states (N, 1, 1, 4)
                best_action_index = torch.argmax(q_values, dim=3, keepdim=True)  # (N, 1, 1, 1)
                best_action_index = torch.flatten(best_action_index)
                return best_action_index.detach().cpu().numpy()  # N entry nparray

    """
    Does a forward call and returns the Q value of an action at a specified index given a state
    and an index. Meant to be used on the target CNN.
    Input: State sequence of (self.seq_size) frames, int index of specified action OR 
    (N) tensor of indices
    Output: Q value of specified action (float OR nparray of floats)
    """
    def get_q_value_of_action(self, state, action_index):
        with torch.no_grad():
            q_values = self.forward(state)  # (N, 4, 200, 200) -> (N, 1, 1, 4)
            if q_values.dim() == 3:  # if single state (1, 1, 4)
                q_values = torch.flatten(q_values)  # (1, 1, 4) -> (4)
                return q_values[action_index].item()  # float
            else:  # if batch of states (N, 1, 1, 4)
                q_values = torch.flatten(q_values, start_dim=1, end_dim=2)  # (N, 4)
                q_values = q_values.detach().cpu().numpy()  # Nx4 nparray
                q_of_actions = q_values[range(q_values.shape[0]), action_index]
                return q_of_actions  # Nx1 nparray of floats


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
                                self.frame_h, self.frame_w), dtype=np.uint8)  # Dim: (32, 4, 200, 200)
        self.new_states = np.empty((self.batch_size, self.seq_size,
                                    self.frame_h, self.frame_w), dtype=np.uint8)  # Dim: (32, 4, 200, 200)
        # List of indices to slice out of the memory
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    # Should be easily called every frame
    def add_memory(self, frame, action, reward, is_terminal):
        self.frames[self.current, ...] = frame
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminals[self.current] = is_terminal
        self.counter = max(self.counter, self.current + 1)
        self.current = (self.current + 1) % self.size

    # A state is composed of four frames (tensor dim: [4, 200, 200])
    def get_state(self, index):
        if self.counter == 0 or index < self.seq_size - 1:
            pass
        else:
            return self.frames[index - self.seq_size + 1: index + 1, ...]

    # Pick random valid indices to slice out of the memory
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

    # Use the randomly picked indices to slice out states and next states to be stored
    # in network fields
    def get_minibatch(self):
        if self.counter < self.seq_size:
            pass
        self.get_indices()
        for i, idx in enumerate(self.indices):
            self.states[i] = self.get_state(idx - 1)
            self.new_states[i] = self.get_state(idx)
        # TODO: Wrong return dimensions. Look at __init__ for dimensions.
        return self.states, self.actions, self.rewards, self.new_states, self.terminals
