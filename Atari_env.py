import pygame
import player
import platforms
import params
import sys
import spritesheet
import time
import itertools
from random import randint
import random
import Q_params
import statistics
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gym
import imageio

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def grayscale(frame):
    tensor_rgb = torch.from_numpy(frame)  # Atari frames: (210, 160, 3)
    tensor_rgb = torch.permute(tensor_rgb, (2, 1, 0))  # (3, 160, 210)
    grayscale_tensor = torchvision.transforms.functional.rgb_to_grayscale(tensor_rgb, 1)  # (1, 160, 210)
    resized_gray = torchvision.transforms.functional.resized_crop(grayscale_tensor, top=34, left=0, height=160,
                                                                  width=160, size=(84, 84))  # (1, 84, 84)
    np_gray = resized_gray.detach().cpu().numpy()
    np_efficient = np_gray.astype(dtype=np.uint8)
    return np_efficient


def norm_reward(reward):
    if reward > 0:
        return 1
    elif reward < 0:
        return -1
    else:
        return 0


def replay(memory, main_network, target_network):
    """
    Performs minibatch sampling from replay memory, sets the Bellman target Q for each step, and
    performs minibatch gradient descent.
    """
    main_network.train()
    torch.set_grad_enabled(True)
    states, actions, rewards, new_states, terminals = memory.get_minibatch()
    argmax_q_main = main_network.get_highest_q_action(new_states)  # size N nparray
    double_q = target_network.get_q_value_of_action(new_states, argmax_q_main)  # Nx1 nparray
    target = rewards + main_network.gamma * double_q * (1 - terminals.astype(int))
    predict = main_network.get_q_value_of_action(states, actions)
    loss = F.huber_loss(input=torch.from_numpy(predict), target=torch.from_numpy(target), reduction='mean',
                        delta=1.0)  # mean reduction
    loss.requires_grad_()
    main_network.optimizer.zero_grad()
    loss.backward()
    main_network.optimizer.step()
    return loss


class Atari(object):
    def __init__(self, env_name, render_game, no_op_steps=10):
        if render_game:
            self.env = gym.make(env_name, render_mode='human')
        else:
            self.env = gym.make(env_name)
        self.seq_size = 4
        self.curr_frame = np.empty((1, 84, 84), dtype=np.uint8)
        self.state = np.empty((self.seq_size, 84, 84), dtype=np.uint8)
        self.last_lives = 0
        self.no_op_steps = no_op_steps

    def new_game(self):
        first_frame = self.env.reset()
        self.last_lives = 0
        terminal_life_lost = True
        grayscale_frame = grayscale(first_frame)
        self.curr_frame = grayscale_frame
        sequence_frames = np.repeat(grayscale_frame, self.seq_size, axis=0)
        self.state = sequence_frames
        return terminal_life_lost

    def step(self, action):
        next_frame, reward, terminal, info = self.env.step(action)
        if info['lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['lives']
        grayscale_frame = grayscale(next_frame)
        self.curr_frame = grayscale_frame
        next_state = np.concatenate((self.state[1:, :, :], grayscale_frame), axis=0)
        self.state = next_state
        return grayscale_frame, reward, terminal, terminal_life_lost


class D3QAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.reward = 0
        self.gamma = 0.99
        self.learning_rate = params['learning_rate']
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']

        # Annealing epsilon
        self.epsilon_init = 1
        self.epsilon_final = 0.1
        self.epsilon_decay = 0.01
        self.replay_start_size = params['replay_start']
        self.anneal_frames = params['anneal_frames']
        self.max_frames = params['max_frames']
        self.slope = -(self.epsilon_init - self.epsilon_final) / self.anneal_frames
        self.intercept = self.epsilon_init - self.slope * self.replay_start_size
        self.slope_2 = -(self.epsilon_final - self.epsilon_decay) / (self.max_frames - self.anneal_frames
                                                                     - self.replay_start_size)
        self.intercept_2 = self.epsilon_decay - self.slope_2 * self.max_frames

        # Convolutional Layers - (N, C_in, H, W) -> (N, C_out, H_out, W_out)
        # Dimension will be: (32, 4, 84, 84) -> (32, C_out, H_out, W_out)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding='valid', bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding='valid', bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='valid', bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=1024, kernel_size=7, stride=1, padding='valid', bias=False)

        # Output Layers - (*, C_in) -> (*, C_out)
        self.value_layer = nn.Linear(512, 1)
        self.adv_layer = nn.Linear(512, 6)

        # Updates
        self.target = None  # Bellman equation target Q
        self.action = None  # Action taken
        self.optimizer = optim.Adam(self.parameters(), weight_decay=0, lr=self.learning_rate)

        # Weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def forward(self, x):
        """
        Forward call through 4 convolution layers and final split linear layer for dueling.

        Input: Grayscale normalized image tensor, mini-batches allowed. Dim: (N, C_in, H, W)

        Output: (N, C_out, H, W) Tensor of q_values for each action
        """
        x = F.relu(self.conv1(x))  # (32, 4, 84, 84) tensor in
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # (32, 1024, 1, 1) tensor out
        if x.dim() == 3:
            conv_value, conv_adv = torch.split(x, 512, dim=0)  # (512, 1, 1)
            conv_value = torch.permute(conv_value, (1, 2, 0))  # (1, 1, 512)
            conv_adv = torch.permute(conv_adv, (1, 2, 0))  # (1, 1, 512)
        else:
            conv_value, conv_adv = torch.split(x, 512, dim=1)  # (32, 512, 1, 1)
            conv_value = torch.permute(conv_value, (0, 2, 3, 1))  # (32, 1, 1, 512)
            conv_adv = torch.permute(conv_adv, (0, 2, 3, 1))  # (32, 1, 1, 512)
        value = self.value_layer(conv_value)  # (32, 1, 1, 512) -> (32, 1, 1, 1)
        adv = self.adv_layer(conv_adv)  # (32, 1, 1, 512) -> (32, 1, 1, 6)
        adv_average = torch.mean(adv, dim=(adv.dim() - 1), keepdim=True)  # (32, 1, 1, 1)
        q_values = torch.subtract(torch.add(adv, value), adv_average)  # broadcast (32, 1, 1, 6)
        return q_values

    def get_highest_q_action(self, state):
        """
        Does a forward call and returns the index of the action with the highest Q value given a
        state. Meant to be used on the main CNN.

        Input: State sequence of (self.seq_size) frames

        Output: Index of best action in action list (int)
        """
        with torch.no_grad():
            q_values = self.forward(state)  # (N, 4, 84, 84) -> (N, 1, 1, 6)
            if state.dim() == 3:  # if single state (1, 1, 6)
                q_values = torch.flatten(q_values)  # (1, 1, 6) -> (6)
                best_action_index = torch.argmax(q_values)  # (1)
                return best_action_index.item()  # int
            else:  # if batch of states (N, 1, 1, 6)
                best_action_index = torch.argmax(q_values, dim=3, keepdim=True)  # (N, 1, 1, 1)
                best_action_index = torch.flatten(best_action_index)  # (N)
                return best_action_index.detach().cpu().numpy()  # size N nparray

    def get_q_value_of_action(self, state, action_index):
        """
        Does a forward call and returns the Q value of an action at a specified index given a state
        and an index. Meant to be used on the target CNN.

        Input: State sequence of (self.seq_size) frames, int index of specified action OR
        size (N,) nparray of indices

        Output: Q value of specified action (float OR nparray of floats)
        """
        with torch.no_grad():
            q_values = self.forward(state)  # (N, 4, 84, 84) -> (N, 1, 1, 6)
            if state.dim() == 3:  # if single state (1, 1, 6)
                q_values = torch.flatten(q_values)  # (1, 1, 6) -> (6)
                return q_values[action_index].item()  # float
            else:  # if batch of states (N, 1, 1, 6)
                q_values = torch.flatten(q_values, start_dim=1, end_dim=3)  # (N, 6)
                q_values = q_values.detach().cpu().numpy()  # Nx2 nparray
                q_of_actions = q_values[range(q_values.shape[0]), action_index.tolist()]
                return q_of_actions  # Nx1 nparray of floats

    def choose_action(self, state, frame_num):
        if frame_num < self.replay_start_size:
            epsilon = self.epsilon_init
        elif self.replay_start_size <= frame_num < self.replay_start_size + self.anneal_frames:
            epsilon = self.slope * frame_num + self.intercept
        else:
            epsilon = self.slope_2 * frame_num + self.intercept_2

        if random.uniform(0, 1) < epsilon:
            return randint(0, 5)
        else:
            return self.get_highest_q_action(state)


class Memory(object):
    def __init__(self, size=500000, frame_h=84, frame_w=84, batch_size=32, seq_size=4):
        self.counter = 0
        self.current = 0
        self.size = size  # computer may have overcommitting problems with large sizes
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.batch_size = batch_size
        self.seq_size = seq_size

        self.frames = np.empty((self.size, self.frame_h, self.frame_w), dtype=np.uint8)
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.terminals = np.empty(self.size, dtype=bool)

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

    # A state is composed of four frames: (4, 84, 84) tensor
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
        if torch.cuda.is_available():
            return torch.div(torch.from_numpy(self.states).cuda(), 255), self.actions[self.indices], self.rewards[
                self.indices], \
                   torch.div(torch.from_numpy(self.new_states).cuda(), 255), self.terminals[self.indices]
        return torch.div(torch.from_numpy(self.states), 255), self.actions[self.indices], self.rewards[self.indices], \
               torch.div(torch.from_numpy(self.new_states), 255), self.terminals[self.indices]


def training():

    q_params = Q_params.params_Q

    # Initialize memory and networks
    replay_memory = Memory()
    main_network = D3QAgent(q_params)
    main_network = main_network.to(DEVICE)
    target_network = D3QAgent(q_params)
    target_network = target_network.to(DEVICE)

    # Initialize tracking quantities
    avg_episode_loss = []
    episode_rewards = []
    frame = 0
    max_frame = q_params['max_frames']
    epoch_max_frame = 100000
    episode_num = 0
    episode_max_frame = 18000

    # Initialize atari game
    # pip3 install gym[atari,accept-rom-license]==0.21.0
    env_name = "ALE/Pong-v5"
    render_game = False
    atari = Atari(env_name, render_game)
    print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                    atari.env.unwrapped.get_action_meanings()))
    try:
        while frame < max_frame:
            epoch_frame = 0
            while epoch_frame < epoch_max_frame:
                terminal_life_lost = atari.new_game()
                episode_reward = 0
                episode_loss = []
                for i in range(episode_max_frame):
                    if torch.cuda.is_available():
                        curr_state = torch.div(torch.from_numpy(atari.state).cuda(), 255)
                    else:
                        curr_state = torch.div(torch.from_numpy(atari.state), 255)
                    action = main_network.choose_action(curr_state, frame)
                    next_frame, reward, terminal, terminal_life_lost = atari.step(action)
                    frame += 1
                    epoch_frame += 1
                    normed_reward = norm_reward(reward)
                    episode_reward += normed_reward
                    replay_memory.add_memory(next_frame, action, normed_reward, terminal_life_lost)

                    # Perform gradient descent
                    loss = 0
                    if frame % q_params['update_frequency'] == 0 and frame > q_params['replay_start']:
                        loss = replay(replay_memory, main_network, target_network)
                        if torch.is_tensor(loss):
                            loss = loss.item()
                    episode_loss.append(loss)

                    # Update target network
                    if frame % q_params['net_update_frequency'] == 0 and frame > q_params['replay_start']:
                        target_network.load_state_dict(main_network.state_dict())

                    if terminal:
                        break

                    if not loss == 0:
                        print(loss)

                episode_num += 1
                episode_rewards.append(episode_reward)
                avg_episode_loss.append(np.mean(episode_loss))
                print(f'Frame {frame}')
                print(f'Game {episode_num}')
                print('Episode Reward: ' + str(episode_reward))
        return episode_num, episode_rewards, avg_episode_loss, main_network.state_dict()
    except KeyboardInterrupt:
        print('Interrupted')
        return episode_num, episode_rewards, avg_episode_loss, main_network.state_dict()


def plotter(episode_num, episode_rewards, avg_episode_loss):
    episodes = np.arange(1, episode_num + 1)
    plt.plot(episodes, avg_episode_loss)
    plt.xlabel('Episode')
    plt.ylabel('Huber Loss')
    plt.show()

    plt.plot(episodes, episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.show()


if __name__ == '__main__':
    if __name__ == '__main__':
        episode_num, episode_rewards, avg_episode_loss, main_network_weights = training()
        torch.save(main_network_weights, 'atari_weights.pt')
        plotter(episode_num, episode_rewards, avg_episode_loss)



