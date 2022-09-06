import pygame
import player
import platforms
import params
import sys
import spritesheet
import time
import itertools
from deepQ import deepQAgent
from random import randint
import random
import statistics
import torch.optim as optim
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

pygame.init()

vec = pygame.math.Vector2  # 2 for two dimensional
FramePerSec = pygame.time.Clock()
brick = pygame.image.load("background.png")
displaysurface = pygame.display.set_mode((params.WIDTH, params.HEIGHT))
pygame.display.set_caption("Game")


def set_background():
    displaysurface.fill(params.WHITE)
    brick_width, brick_height = brick.get_width(), brick.get_height()
    for x, y in itertools.product(range(0, params.WIDTH, brick_width), range(0, params.HEIGHT, brick_height)):
        displaysurface.blit(brick, (x, y))


# Initiate player and starting platform
spider = player.Player()
plat_image = pygame.image.load('wood_platform.png')
plat_image.set_colorkey((0,0,0))
platform1 = platforms.Platform()
platform1.surf = pygame.transform.scale(plat_image, (params.WIDTH, 20))
platform1.rect = platform1.surf.get_rect(center=(params.WIDTH / 2, params.HEIGHT - 10))
platform1.moving = False
platform1.point = False

# Collection of all sprites on screen
all_sprites = pygame.sprite.Group()
all_sprites.add(spider)
all_sprites.add(platform1)

# Separate platform sprite group
plats = pygame.sprite.Group()
plats.add(platform1)

# Initialize starting screen random platforms
for x in range(random.randint(5, 6)):
    pl = platforms.Platform()
    close = True
    while close:
        pl = platforms.Platform()
        close = platforms.check(pl, plats)
    plats.add(pl)
    all_sprites.add(pl)

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