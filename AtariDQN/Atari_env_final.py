import random
import statistics

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from Atari_wrappers import *

cv2.ocl.setUseOpenCL(False)


def init_params():
    q_params = dict()
    q_params['env_name'] = "PongNoFrameskip-v4"
    q_params['render_game'] = False
    q_params['seed'] = 42
    q_params['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    q_params['learning_rate'] = 0.0001
    q_params['batch_size'] = 32
    q_params['gamma'] = 0.99
    q_params['m_update_frequency'] = 1
    q_params['t_update_frequency'] = 1000
    q_params['memory_size'] = 5000
    q_params['replay_start'] = 10000
    q_params['max_frames'] = 1000000
    q_params['epsilon_init'] = 1
    q_params['epsilon_final'] = 0.01
    q_params['e_scale_factor'] = 0.1
    q_params['weights_path'] = 'atari_weights.pt'
    q_params['load_weights'] = False

    np.random.seed(q_params["seed"])
    random.seed(q_params["seed"])
    return q_params


class Memory(object):
    def __init__(self, size, batch_size):
        self.max_size = size
        self.current = 0
        self.memory = []

    def add_memory(self, state, action, reward, next_state, terminal):
        data = (state, action, reward, next_state, terminal)
        # if out of memory, start replacing earlier transitions
        if self.current >= len(self.memory):
            self.memory.append(data)
        else:
            self.memory[self.current] = data
        self.current = (self.current + 1) % self.max_size


    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.memory) - 1, size=batch_size)
        states, actions, rewards, next_states, terminals = [], [], [], [], []
        for i in indices:
            data = self.memory[i]
            state, action, reward, next_state, terminal = data
            states.append(np.array(state))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state))
            terminals.append(terminal)
        return np.array(states) / 255, np.array(actions), np.array(rewards), np.array(next_states) / 255, np.array(terminals)


class Atari(object):
    def __init__(self, params):
        if params['render_game']:
            self.env = gym.make(params['env_name'], render_mode='human')
        else:
            self.env = gym.make(params['env_name'])
        self.env.seed(params["seed"])
        self.env = NoopResetEnv(self.env, noop_max=30)
        self.env = MaxAndSkipEnv(self.env, skip=4)
        self.env = EpisodicLifeEnv(self.env)
        self.env = FireResetEnv(self.env)
        self.env = WarpFrame(self.env)
        self.env = PyTorchFrame(self.env)
        self.env = ClipRewardEnv(self.env)
        self.env = FrameStack(self.env, 4)


class D2QStruct(torch.nn.Module):
    def __init__(self, atari):
        super().__init__()
        observation_space = atari.env.observation_space
        action_space = atari.env.action_space

        # Convolutional Layers
        self.convoluted = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()
        )

        # Fully Connected Layers
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=32 * 9 * 9, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_space.n)
        )

    def forward(self, x):
        conv_out = self.convoluted(x)
        conv_out = conv_out.reshape(x.size()[0], -1)
        return self.fully_connected(conv_out)


class D2QAgent:
    def __init__(self, atari, params):
        self.memory = Memory(params['memory_size'], params['batch_size'])
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.device = params['device']

        self.main_network = D2QStruct(atari).to(self.device)
        self.target_network = D2QStruct(atari).to(self.device)
        self.update_target_network()
        self.target_network.eval()

        self.optimizer = torch.optim.RMSprop(self.main_network.parameters(), lr=params['learning_rate'])

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def replay(self):
        device = self.device
        states, actions, rewards, next_states, terminals = self.memory.sample(self.batch_size)
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        terminals = torch.from_numpy(terminals).float().to(device)

        with torch.no_grad():
            _, max_next_action = torch.max(self.main_network(next_states), dim=1)
            max_next_q_values = torch.gather(self.target_network(next_states), 1, max_next_action.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - terminals) * self.gamma * max_next_q_values

        predict_q_values = self.main_network(states)
        predict_q_values = predict_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(predict_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del states
        del next_states
        return loss.item()

    def choose_action(self, state):
        device = self.device
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.main_network(state)
            _, action = torch.max(q_values, dim=1)
            return action.item()


def training():
    # Initialize parameters
    q_params = init_params()

    # Initialize atari game
    # pip3 install gym[atari,accept-rom-license]==0.21.0
    atari = Atari(q_params)
    print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                    atari.env.unwrapped.get_action_meanings()))

    # Initialize double q network
    d2q = D2QAgent(atari, q_params)

    # Initialize tracking quantities
    avg_episode_loss = [0.0]
    episode_losses = [0.0]
    episode_rewards = [0.0]
    epsilon_frames = q_params['e_scale_factor'] * float(q_params['max_frames'])

    try:
        state = atari.env.reset()
        for frame in range(q_params['max_frames']):
            fraction = min(1.0, float(frame) / epsilon_frames)
            epsilon = q_params['epsilon_init'] + fraction * \
                            (q_params['epsilon_final'] - q_params['epsilon_init'])

            prob = random.random()
            if prob < epsilon:
                action = atari.env.action_space.sample()
            else:
                action = d2q.choose_action(state)

            next_state, reward, terminal, info = atari.env.step(action)
            d2q.memory.add_memory(state, action, reward, next_state, float(terminal))
            state = next_state
            episode_rewards[-1] += reward

            if terminal:
                state = atari.env.reset()
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
                loss = d2q.replay()
                episode_losses.append(loss)

            if frame > q_params['replay_start'] and frame % q_params['t_update_frequency'] == 0:
                d2q.update_target_network()
        return episode_rewards, avg_episode_loss, d2q.main_network.state_dict()

    except KeyboardInterrupt:
        print('Interrupted!')
        return episode_rewards, avg_episode_loss, d2q.main_network.state_dict()


def plotter(episode_rewards, avg_episode_loss):
    episodes = np.arange(1, len(episode_rewards) + 1)
    plt.plot(episodes, avg_episode_loss)
    plt.xlabel('Episode')
    plt.ylabel('Smooth L1 Loss')
    plt.show()

    plt.plot(episodes, episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.show()


if __name__ == '__main__':
    episode_rewards, avg_episode_loss, main_network_weights = training()
    torch.save(main_network_weights, 'atari_weights.pt')
    plotter(episode_rewards, avg_episode_loss)



