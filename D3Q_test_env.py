import pygame
import D3Q
import player
import platforms
import params
import sys
import spritesheet
import time
import itertools
from deepQ import DeepQAgent
from random import randint
import random
import Q_params
import statistics
import torch.optim as optim
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SpiderJumpGame:
    def __init__(self):
        pygame.display.set_caption("SpiderJump")
        self.displaysurface = pygame.display.set_mode((params.WIDTH, params.HEIGHT))
        self.spider = player.Player()
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.spider)
        self.plats = pygame.sprite.Group()
        self.play_plats = pygame.sprite.Group()
        self.plat_image = pygame.image.load('wood_platform.png')
        self.plat_image.set_colorkey((0, 0, 0))
        self.background = pygame.image.load("background.png")
        self.FramePerSec = pygame.time.Clock()
        self.font_type = pygame.font.SysFont("Verdana", 20)
        self.game_over = False


def set_background(displaysurface, background):
    displaysurface.fill(params.WHITE)
    bg_width, bg_height = background.get_width(), background.get_height()
    for x, y in itertools.product(range(0, params.WIDTH, bg_width), range(0, params.HEIGHT, bg_height)):
        displaysurface.blit(background, (x, y))


def set_start_plats(plat_image, plats, all_sprites, play_plats):
    platform1 = platforms.Platform()
    platform1.surf = pygame.transform.scale(plat_image, (params.WIDTH, 20))
    platform1.rect = platform1.surf.get_rect(center=(params.WIDTH / 2, params.HEIGHT - 10))
    platform1.moving = False
    platform1.point = False
    all_sprites.add(platform1)
    plats.add(platform1)
    for x in range(random.randint(5, 6)):
        pl = platforms.Platform()
        close = True
        while close:
            pl = platforms.Platform()
            close = platforms.check(pl, plats)
        play_plats.add(pl)
        plats.add(pl)
        all_sprites.add(pl)


def build_start(game):
    set_background(game.displaysurface, game.background)
    set_start_plats(game.plat_image, game.plats, game.all_sprites, game.play_plats)


# 4 possible action states: left, right, jump, release jump
def do_action(game, action):
    if action[0] == 1 or action[1] == 1:
        game.spider.agent_move(action)
    elif action[2] == 1:
        game.spider.jump(game.plats)
    else:
        game.spider.release_jump(game.plats)


"""
Obtains an RGB image of the current game screen, then grayscales and downsizes it to a 200 x 200
np uint8 array for memory efficiency.
Input: game object
Output: grayscale np array (dtype = uint8) of current game frame
"""
def to_grayscale(game):
    screen_area = pygame.Rect(0, params.HEIGHT, params.WIDTH, params.HEIGHT)
    sub_surface = game.displaysurface.subsurface(screen_area)
    RGB_data = pygame.surfarray.array3d(sub_surface)
    tensor_RGB = torch.from_numpy(RGB_data)
    grayscale_tensor = torchvision.transforms.functional.rgb_to_grayscale(tensor_RGB, 1)
    resized_gray = torchvision.transforms.functional.resized_crop(grayscale_tensor, top=400, left=0, height=400,
                                                                  width=400, size=(200, 200))
    # normalized_gray = torch.div(resized_gray, 255)
    np_gray = resized_gray.detach().cpu().numpy()
    np_efficient = np_gray.astype(dtype=np.uint8)
    return np_efficient


"""
Initializes the first sequence: Updates the agent.input field by cloning the first frame of
the game into a (seq-size)-layer 3-dim tensor to be used for the first forward.
"""
def init_sequence(game, agent, seq_size):
    first_frame = to_grayscale(game)
    sequence_frames = np.repeat(first_frame, seq_size, axis=2)
    sequence_tensor = torch.from_numpy(sequence_frames)
    agent.input = torch.div(sequence_tensor, 255)


# ???
def update_sequence(game, agent):
    curr_frame = torch.from_numpy(to_grayscale(game))
    norm_frame = torch.div(curr_frame, 255)
    curr_sequence = agent.input
    new_sequence = torch.stack((curr_sequence[:, :, 1:], norm_frame), dim=2)
    agent.input = new_sequence


"""
Performs minibatch sampling from replay memory, sets the Bellman target Q for each step, and
performs gradient descent.
"""
def train_short(memory, main_network, target_network):
    states, actions, rewards, new_states, terminals = memory.get_minibatch()
    for i in range(memory.batch_size):  # ???? for now all q's are single int/floats
        argmax_q_main = main_network.get_highest_q_action(new_states[i])
        double_q = target_network.get_q_value_of_action(new_states[i], argmax_q_main)
        main_network.target = rewards[i] + main_network.gamma * double_q * (1 - int(terminals[i]))  # Bellman eq
        inputt = 1  # what is the input ??? Q(s_j, a_j) implies Q value of state-action leading to next state
        loss = F.huber_loss(input=inputt, target=main_network.target, reduction='mean', delta=1.0)
        main_network.optimizer.zero_grad()
        loss.backward()
        main_network.optimizer.step()


# Reference https://github.com/gouxiangchen/dueling-DQN-pytorch/blob/master/dueling_dqn.py
def training():
    pygame.init()
    q_params = Q_params.params_Q

    replay_memory = D3Q.Memory()
    main_network = D3Q.D3QAgent(q_params)
    target_network = D3Q.D3QAgent(q_params)
    frame = 0
    max_frame = 100000000
    while frame < max_frame:
        # init game and sequence
        game = SpiderJumpGame()
        build_start(game)
        init_sequence(game, main_network)
        # while gaming do
            # select action based on epsilon or network
            # observe frame
            # do action and observe reward
            # observe next frame
            # update memory frames and sequence frames
            # if time to learn
                # train_short
            # if time to update target network
                # update target network