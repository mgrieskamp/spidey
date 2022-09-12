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
import torchvision
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


def play():
    pygame.init()
    game = SpiderJumpGame()
    build_start(game)
    running = True
    while running:
        set_background(game.displaysurface, game.background)
        game.spider.update(game.plats, game.play_plats)
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


# 4 possible action states: left, right, jump, release jump
def do_action(game, action):
    if action[0] == 1 or action[1] == 1:
        game.spider.agent_move(action)
    elif action[2] == 1:
        game.spider.jump(game.plats)
    else:
        game.spider.release_jump(game.plats)


# Initializes the agent by performing one action and one transition in the game world
def init_agent(game, agent, batch_size):
    init_state1 = agent.get_state(game.spider, game.play_plats)
    action = [1, 0, 0, 0]
    do_action(game, action)
    game.spider.update(game.plats, game.play_plats)  ##new
    init_state2 = agent.get_state(game.spider, game.play_plats)
    init_reward = agent.set_reward(game.spider, game.game_over, init_state1)
    agent.store_transition(init_state1, action, init_reward, init_state2, game.game_over)
    agent.replay_memory(agent.memory, batch_size)


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


def update_sequence(game, agent):
    curr_frame = torch.from_numpy(to_grayscale(game))
    norm_frame = torch.div(curr_frame, 255)
    curr_sequence = agent.input
    new_sequence = torch.stack((curr_sequence[:, :, 1:], norm_frame), dim=2)
    agent.input = new_sequence


def train_new():
    pygame.init()
    frame_num = 0
    max_frames = 10000000
    while frame_num < max_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


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
        episode_reward = 0

        # play until game over or no progress for 500 steps
        while (not game.game_over) and (steps < 500):

            set_background(game.displaysurface, game.background)
            game.spider.update(game.plats, game.play_plats)  ##new

            if q_params['train']:
                # epsilon (random exploration) decreases as agent trains for longer
                agent.epsilon = 1 - (num_games * q_params['epsilon_decay_linear'])
            else:
                agent.epsilon = 0.01

            curr_state = agent.get_state(game.spider, game.play_plats)

            # perform random action based on epsilon (explore) or choose action based on Q function prediction
            curr_action = [0, 0, 0, 0, 0]
            if random.uniform(0, 1) < agent.epsilon:
                curr_action[randint(0, 4)] = 1
            else:
                with torch.no_grad():
                    curr_state_tensor = torch.from_numpy(curr_state)  # cuda() before every forward?
                    pred = agent.forward(curr_state_tensor.float())  # float?
                    ''' TODO: unsure if this will return an action between 0 and 4, I think we need to update our neural net
                    to have correct input and output layer sizes (input=36 output=5) '''
                    curr_action[np.argmax(pred.detach().cpu().numpy())] = 1
            do_action(game, curr_action)

            if game.spider.rect.bottom > params.HEIGHT:
                game.game_over = True
                game.spider.kill()

            if game.spider.rect.top <= params.HEIGHT / 3:
                game.spider.pos.y += abs(game.spider.vel.y)
                for pl in game.plats:
                    pl.rect.y += abs(game.spider.vel.y)
                    if pl.rect.top > params.HEIGHT:
                        pl.kill()

            platforms.plat_gen(game.plats, game.all_sprites, game.play_plats)

            game_score = game.font_type.render(str(game.spider.score), True, (123, 255, 0))
            game.displaysurface.blit(game_score, (params.WIDTH / 2, 10))

            for entity in game.all_sprites:
                entity.draw(game.displaysurface)
                entity.move()

            pygame.display.update()
            game.FramePerSec.tick(params.FPS)

            next_state = agent.get_state(game.spider, game.play_plats)
            reward = agent.set_reward(game.spider, game.game_over, curr_state)
            episode_reward += reward

            # if spider landed on platform, reset steps
            if reward > 0:
                steps = 0

            if q_params['train']:
                agent.train_short_term(curr_state, curr_action, reward, next_state, game.game_over)
                agent.store_transition(curr_state, curr_action, reward, next_state, game.game_over)

            steps += 1
            # print(steps)

        num_games += 1
        total_score += game.spider.score
        print(f'Game {num_games}        Score: {game.spider.score}')
        print('Episode Reward: ' + str(episode_reward))
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
