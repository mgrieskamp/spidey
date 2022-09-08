import pygame
import deepQ
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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'


class SpiderJumpGame:
    def __init__(self):
        pygame.display.set_caption("SpiderJump")
        self.displaysurface = pygame.display.set_mode((params.WIDTH, params.HEIGHT))
        self.spider = player.Player()
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.spider)
        self.plats = pygame.sprite.Group()
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


def set_start_plats(plat_image, plats, all_sprites):
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
        plats.add(pl)
        all_sprites.add(pl)


def build_start(game):
    set_background(game.displaysurface, game.background)
    set_start_plats(game.plat_image, game.plats, game.all_sprites)


def play():
    pygame.init()
    game = SpiderJumpGame()
    build_start(game)
    running = True
    while running:
        set_background(game.displaysurface, game.background)
        game.spider.update(game.plats)
        # Track player inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Exit the while loop
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.spider.jump(game.plats)
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    game.spider.release_jump(game.plats)
        # Initiate game over once the player falls off the screen
        if game.spider.rect.top > params.HEIGHT:
            game.game_over = True
            for sprite in game.all_sprites:
                sprite.kill()
                time.sleep(1)
                game.displaysurface.fill(params.WHITE)
                pygame.display.update()
                time.sleep(1)
                pygame.quit()
                sys.exit()
        # Screen moving (in y-axis) and kills bottom platforms
        if game.spider.rect.top <= params.HEIGHT / 3:
            game.spider.pos.y += abs(game.spider.vel.y)
            for pl in game.plats:
                pl.rect.y += abs(game.spider.vel.y)
                if pl.rect.top > params.HEIGHT:
                    pl.kill()
        # Generate new random platforms as player moves up
        platforms.plat_gen(game.plats, game.all_sprites)
        # Display game score
        game_score = game.font_type.render(str(game.spider.score), True, (123, 255, 0))
        game.displaysurface.blit(game_score, (params.WIDTH / 2, 10))
        # Loops through all sprites on screen
        for entity in game.all_sprites:
            entity.draw(game.displaysurface)
            entity.move()
        pygame.display.update()
        game.FramePerSec.tick(params.FPS)
    pygame.quit()
    sys.exit()


def do_action(game, action):
    if action[0] == 1 or action[1] == 1:
        game.spider.agent_move(action)
    elif action[2] == 1:
        game.spider.jump(game.plats)
    elif action[3] == 1:
        game.spider.release_jump(game.plats)
    else:
        pass


def init_agent(game, agent, batch_size):
    init_state1 = agent.get_state(game.spider, game.plats)
    action = [1, 0, 0, 0, 0]
    do_action(game, action)
    game.spider.update(game.plats) ##new
    init_state2 = agent.get_state(game.spider, game.plats)
    init_reward = agent.set_reward(game.spider, game.game_over)
    agent.store_transition(init_state1, action, init_reward, init_state2, game.game_over)
    agent.replay_memory(agent.memory, batch_size)


def train_Q():
    pygame.init()
    q_params = Q_params.params_Q
    num_games = 0
    scores = []
    counter = []
    record = 0
    total_score = 0
    agent = deepQ.DeepQAgent(q_params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=q_params['learning_rate'])
    # start game loop
    while num_games < q_params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        game = SpiderJumpGame()
        build_start(game)

        # first move
        init_agent(game, agent, q_params['batch_size'])
        steps = 0

        # play until game over or no progress for 1000 steps
        while (not game.game_over) and (steps < 1000):

            set_background(game.displaysurface, game.background) ##new
            game.spider.update(game.plats) ##new

            if q_params['train']:
                # epsilon (random exploration) decreases as agent trains for longer
                agent.epsilon = 1 - (num_games * q_params['epsilon_decay_linear'])
            else:
                agent.epsilon = 0.01

            curr_state = agent.get_state(game.spider, game.plats)

            # perform random action based on epsilon (explore) or choose action based on Q function prediction
            curr_action = [0, 0, 0, 0, 0]
            if random.uniform(0, 1) < agent.epsilon:
                curr_action[randint(0, 4)] = 1
            else:
                with torch.no_grad():
                    curr_state_tensor = torch.from_numpy(curr_state)
                    pred = agent.forward(curr_state_tensor.float()) #float?
                    ''' TODO: unsure if this will return an action between 0 and 4, I think we need to update our neural net
                    to have correct input and output layer sizes (input=36 output=5) '''
                    curr_action[np.argmax(pred.detach().cpu().numpy())] = 1
            do_action(game, curr_action)

            if game.spider.rect.top > params.HEIGHT:
                game.game_over = True

            if game.spider.rect.top <= params.HEIGHT / 3:
                game.spider.pos.y += abs(game.spider.vel.y)
                for pl in game.plats:
                    pl.rect.y += abs(game.spider.vel.y)
                    if pl.rect.top > params.HEIGHT:
                        pl.kill()

            platforms.plat_gen(game.plats, game.all_sprites)

            game_score = game.font_type.render(str(game.spider.score), True, (123, 255, 0))
            game.displaysurface.blit(game_score, (params.WIDTH / 2, 10))

            for entity in game.all_sprites:
                entity.draw(game.displaysurface)
                entity.move()

            pygame.display.update()
            game.FramePerSec.tick(params.FPS)

            next_state = agent.get_state(game.spider, game.plats)
            reward = agent.set_reward(game.spider, game.game_over)

            # if spider landed on platform, reset steps
            if reward > 0:
                steps = 0

            if q_params['train']:
                agent.train_short_term(curr_state, curr_action, reward, next_state, game.game_over)
                agent.store_transition(curr_state, curr_action, reward, next_state, game.game_over)

            steps += 1
            print(steps)

        num_games += 1
        total_score += game.spider.score
        print(f'Game {num_games}        Score: {game.spider.score}')
        scores.append(game.spider.score)
        counter.append(num_games)

        # replay experiences
        if q_params['train']:
            agent.replay_memory(agent.memory, q_params['batch_size'])

    # TODO: get mean and std. dev. of scores plot
    # update model
    if q_params['train']:
        weights = agent.state_dict()
        torch.save(weights, params['weights_path'])
    if q_params['plot_score']:
        pass
    # TODO: call plotting function
    return total_score


if __name__ == '__main__':
    # play()
    train_Q()

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