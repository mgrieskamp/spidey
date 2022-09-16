from random import randint
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import statistics
import cv2
cv2.ocl.setUseOpenCL(False)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def init_params():
    q_params = dict()
    q_params['env_name'] = "PongNoFrameskip-v4"
    q_params['render_game'] = False
    q_params['seed'] = 31
    q_params['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    q_params['stack_size'] = 4
    q_params['learning_rate'] = 0.00001
    q_params['batch_size'] = 64
    q_params['gamma'] = 0.99
    q_params['tau'] = 0.001
    q_params['m_update_frequency'] = 1
    q_params['t_update_frequency'] = 1
    q_params['memory_size'] = 400000
    q_params['replay_start'] = 40000
    q_params['max_frames'] = 2000000
    q_params['epsilon_init'] = 1
    q_params['epsilon_min'] = 0.01
    q_params['epsilon_step'] = (q_params['epsilon_init'] - q_params['epsilon_min']) / \
                               (q_params['max_frames'] * 0.2)
    q_params['weights_path'] = 'atari_weights_3.pt'
    q_params['load_weights'] = False

    np.random.seed(q_params["seed"])
    random.seed(q_params["seed"])
    return q_params


def grayscale(frame):
    tensor_rgb = torch.from_numpy(frame)  # Atari frames: (210, 160, 3)
    tensor_rgb = torch.permute(tensor_rgb, (2, 1, 0))  # (3, 160, 210)
    grayscale_tensor = torchvision.transforms.functional.rgb_to_grayscale(tensor_rgb, 1)  # (1, 160, 210)
    resized_gray = torchvision.transforms.functional.resized_crop(grayscale_tensor, top=34, left=0, height=160,
                                                                  width=160, size=(84, 84))  # (1, 84, 84)
    np_gray = resized_gray.detach().cpu().numpy()
    np_efficient = np_gray.astype(dtype=np.uint8)
    return np_efficient


class Atari(object):
    def __init__(self, params):
        if params['render_game']:
            self.env = gym.make(params['env_name'], render_mode='human', repeat_action_probability=0.0)
        else:
            self.env = gym.make(params['env_name'], repeat_action_probability=0.0)
        self.env.seed(params['seed'])
        self.seq_size = params['stack_size']
        self.curr_frame = np.empty((1, 84, 84), dtype=np.uint8)
        self.state = np.empty((self.seq_size, 84, 84), dtype=np.uint8)
        self.initialized = False

    def new_game(self):
        first_frame = self.env.reset()
        grayscale_frame = grayscale(first_frame)
        self.curr_frame = grayscale_frame
        if not self.initialized:
            next_state = np.repeat(grayscale_frame, self.seq_size, axis=0)
            self.initialized = True
        else:
            next_state = np.concatenate((self.state[1:, :, :], grayscale_frame), axis=0)
        self.state = next_state

    def step(self, action):
        next_frame, reward, terminal, info = self.env.step(action)
        grayscale_frame = grayscale(next_frame)
        self.curr_frame = grayscale_frame
        next_state = np.concatenate((self.state[1:, :, :], grayscale_frame), axis=0)
        self.state = next_state
        return grayscale_frame, reward, terminal


class D3QStruct(torch.nn.Module):
    def __init__(self, atari):
        super().__init__()
        action_space = atari.env.action_space

        # Convolutional Layers - (N, C_in, H, W) -> (N, C_out, H_out, W_out)
        # Dimension will be: (32, 4, 84, 84) -> (32, C_out, H_out, W_out)
        self.conv_layer_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_layer_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_layer = nn.Linear(7 * 7 * 64, 512)
        # V(s) value of the state
        self.dueling_value = nn.Linear(512, 1)
        # Q(s,a) Q values of the state-action combination
        self.dueling_action = nn.Linear(512, action_space.n)

    def forward(self, x):
        """
        Forward call through 4 convolution layers and final split linear layer for dueling.

        Input: Grayscale normalized image tensor, mini-batches allowed. Dim: (N, C_in, H, W)

        Output: (N, C_out, H, W) Tensor of q_values for each action
        """
        x = F.relu(self.conv_layer_1(x))
        x = F.relu(self.conv_layer_2(x))
        x = F.relu(self.conv_layer_3(x))
        x = F.relu(self.fc_layer(x.view(x.size(0), -1)))
        # get advantage by subtracting dueling action mean from dueling action
        # then add estimated state value
        q_values = self.dueling_action(x) - self.dueling_action(x).mean(dim=1, keepdim=True) + self.dueling_value(x)
        return q_values


class D3QAgent:
    def __init__(self, atari, params):
        self.memory = Memory(params)
        self.tau = params['tau']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.device = params['device']
        self.main_network = D3QStruct(atari).to(self.device)
        self.target_network = D3QStruct(atari).to(self.device)
        self.update_target_network()
        self.target_network.eval()
        self.optimizer = torch.optim.AdamW(self.main_network.parameters(),
                                           lr=params['learning_rate'])

    def update_target_network(self):
        for target_var, var in zip(self.target_network.parameters(), self.main_network.parameters()):
            target_var.data.copy_((1.0 - self.tau) * target_var.data + self.tau * var.data)

    def compute_loss(self):
        device = self.device
        states, actions, rewards, next_states, terminals = self.memory.get_minibatch()
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        terminals = torch.from_numpy(terminals).float().to(device)

        with torch.no_grad():
            _, max_next_action = self.main_network(next_states).max(1)
            max_next_q_values = self.target_network(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - terminals) * self.gamma * max_next_q_values

        input_q_values = self.main_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def choose_action(self, state):
        device = self.device
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.main_network(state)
            _, action = q_values.max(1)
            return action.item()


class Memory(object):
    def __init__(self, params, frame_h=84, frame_w=84):
        self.counter = 0
        self.current = 0
        self.size = params['memory_size']  # computer may have overcommitting problems with large sizes
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.batch_size = params['batch_size']
        self.seq_size = params['stack_size']

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

        return self.states, self.actions[self.indices], self.rewards[self.indices], \
            self.new_states, self.terminals[self.indices]


def training():

    q_params = init_params()

    # Initialize memory and networks
    # Initialize atari game
    # pip3 install gym[atari,accept-rom-license]==0.21.0
    atari = Atari(q_params)
    print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                    atari.env.unwrapped.get_action_meanings()))

    # Initialize dueling double q network
    d3q = D3QAgent(atari, q_params)

    # Initialize tracking quantities
    avg_episode_loss = [0.0]
    episode_losses = [0.0]
    episode_rewards = [0.0]

    try:
        atari.new_game()
        epsilon = q_params['epsilon_init']
        for frame in range(q_params['max_frames']):
            epsilon -= q_params['epsilon_step']
            if epsilon < q_params['epsilon_min']:
                epsilon = q_params['epsilon_min']

            if random.uniform(0, 1) < epsilon:
                action = atari.env.action_space.sample()
            else:
                action = d3q.choose_action(atari.state)

            next_frame, reward, terminal = atari.step(action)
            d3q.memory.add_memory(next_frame, action, reward, terminal)
            episode_rewards[-1] += reward

            if terminal:
                atari.new_game()
                avg_episode_loss[-1] += statistics.mean(episode_losses)
                print('------------------------')
                print('Frame: ' + str(frame))
                print('Episode: ' + str(len(episode_rewards)))
                print('Episode Reward: ' + str(episode_rewards[-1]))
                print('Episode Avg Loss: ' + str(avg_episode_loss[-1]))
                episode_losses = [0.0]
                avg_episode_loss.append(0.0)
                episode_rewards.append(0.0)

            if frame > q_params['replay_start'] and frame % q_params['m_update_frequency'] == 0:
                loss = d3q.compute_loss()
                episode_losses.append(loss)

            if frame > q_params['replay_start'] and frame % q_params['t_update_frequency'] == 0:
                d3q.update_target_network()

        return episode_rewards, avg_episode_loss, d3q.main_network.state_dict()
    except KeyboardInterrupt:
        print('Interrupted!')
        return episode_rewards, avg_episode_loss, d3q.main_network.state_dict()


def plotter(episode_rewards, avg_episode_loss):
    episodes = np.arange(1, len(episode_rewards) + 1)
    plt.plot(episodes, avg_episode_loss)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(episodes, episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.show()


if __name__ == '__main__':
        episode_rewards, avg_episode_loss, main_network_weights = training()
        torch.save(main_network_weights, 'atari_weights_3.pt')
        plotter(episode_rewards, avg_episode_loss)



