import pygame
from deepQ import deepQAgent
from random import randint
import random
import statistics
import torch.optim as optim
import torch

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