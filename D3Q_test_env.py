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
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'


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
        self.rng = np.random.default_rng(seed=2023)


def set_background(displaysurface, background):
    displaysurface.fill(params.WHITE)
    bg_width, bg_height = background.get_width(), background.get_height()
    for x, y in itertools.product(range(0, params.WIDTH, bg_width), range(0, params.HEIGHT, bg_height)):
        displaysurface.blit(background, (x, y))


def set_start_plats(plat_image, plats, all_sprites, play_plats, rng):
    platform1 = platforms.Platform(rng)
    platform1.surf = pygame.transform.scale(plat_image, (params.WIDTH, 20))
    platform1.rect = platform1.surf.get_rect(center=(params.WIDTH / 2, params.HEIGHT - 10))
    platform1.moving = False
    platform1.point = False
    all_sprites.add(platform1)
    plats.add(platform1)
    for x in range(7):
        pl = platforms.Platform(rng)
        close = True
        while close:
            pl = platforms.Platform(rng)
            close = platforms.check(pl, plats)
        play_plats.add(pl)
        plats.add(pl)
        all_sprites.add(pl)


def build_start(game, rng):
    set_background(game.displaysurface, game.background)
    set_start_plats(game.plat_image, game.plats, game.all_sprites, game.play_plats, rng)


# 4 possible action states: left, right, jump, release jump
def do_action(game, action):
    if action[0] == 1 or action[1] == 1:
        game.spider.agent_move(action)
    elif action[2] == 1:
        game.spider.jump(game.plats)
    else:
        game.spider.release_jump(game.plats)


def to_grayscale(game):
    """
    Obtains an RGB image of the current game screen, then grayscales and downsizes it to a 200 x 200
    np uint8 array for memory efficiency.
    @ Input: game object
    @ Output: grayscale np array (dtype = uint8) of current game frame
    """
    screen_area = pygame.Rect(0, params.HEIGHT, params.WIDTH, params.HEIGHT)
    sub_surface = game.displaysurface.subsurface(game.displaysurface.get_rect())
    RGB_data = pygame.surfarray.array3d(sub_surface)
    tensor_RGB = torch.from_numpy(RGB_data)  # (400, 450, 3): (W, H, C)
    tensor_RGB = torch.permute(tensor_RGB, (2, 1, 0))
    grayscale_tensor = torchvision.transforms.functional.rgb_to_grayscale(tensor_RGB, 1)
    resized_gray = torchvision.transforms.functional.resized_crop(grayscale_tensor, top=400, left=0, height=400,
                                                                  width=400, size=(200, 200))
    # normalized_gray = torch.div(resized_gray, 255)
    np_gray = resized_gray.detach().cpu().numpy()
    np_efficient = np_gray.astype(dtype=np.uint8)
    return np_efficient


def init_sequence(game, agent, seq_size):
    """
    Initializes the first sequence: Updates the agent.input field by cloning the first frame of
    the game into a (seq-size)-layer 3-dim tensor to be used for the first forward.
    """
    first_frame = to_grayscale(game)
    sequence_frames = np.repeat(first_frame, seq_size, axis=0)
    sequence_tensor = torch.from_numpy(sequence_frames)
    agent.input = torch.div(sequence_tensor, 255)


# ??? might need reworking
def update_sequence(game, agent):
    curr_frame = torch.from_numpy(to_grayscale(game))
    norm_frame = torch.div(curr_frame, 255)
    curr_sequence = agent.input
    new_sequence = torch.cat((curr_sequence[1:, :, :], norm_frame), dim=0)
    # print(new_sequence.size())
    agent.input = new_sequence


def replay(memory, main_network, target_network):
    """
    Performs minibatch sampling from replay memory, sets the Bellman target Q for each step, and
    performs minibatch gradient descent.
    """
    main_network.train()
    torch.set_grad_enabled(True)
    states, actions, rewards, new_states, terminals = memory.get_minibatch()
    # print(new_states.size())  # good
    # print(new_states.dim())  # good
    argmax_q_main = main_network.get_highest_q_action(new_states)  # size N nparray
    # print("armgax: " + str(argmax_q_main.shape))  # good (32,)
    double_q = target_network.get_q_value_of_action(new_states, argmax_q_main)  # Nx1 nparray
    # print("double_q: " + str(double_q.shape))  # good
    # print("rewards: " + str(rewards.shape))  # good
    # print("terminals: " + str(terminals.shape))  # good
    target = rewards + main_network.gamma * double_q * (1 - terminals.astype(int))
    # print("target: ", target)
    # print("actions: ", actions)
    # predict = torch.sum(torch.multiply(main_network.forward(states), torch.nn.functional.one_hot(torch.from_numpy(actions).long(), 4)))
    predict = main_network.get_q_value_of_action(states, actions)
    # print("predict: ", predict)
    loss = F.huber_loss(input=torch.from_numpy(predict), target=torch.from_numpy(target), reduction='mean', delta=1.0)  # mean reduction
    print("loss: ", loss)
    loss.requires_grad_()
    main_network.optimizer.zero_grad()
    loss.backward()
    main_network.optimizer.step()
    return loss


# Reference https://github.com/gouxiangchen/dueling-DQN-pytorch/blob/master/dueling_dqn.py
# Following the algorithm
def training():
    pygame.init()
    q_params = Q_params.params_Q

    replay_memory = D3Q.Memory()
    main_network = D3Q.D3QAgent(q_params)
    main_network = main_network.to(DEVICE)
    target_network = D3Q.D3QAgent(q_params)
    target_network = target_network.to(DEVICE)

    scores = []
    losses = []
    counter = []
    frame = 0
    max_frame = 100000000
    episode_reward = 0
    episode = 0
    while frame < max_frame:
        episode_frame = 0
        while episode < q_params['episodes']:
            # while gaming do
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if q_params['train']:
                        weights = main_network.state_dict()
                        torch.save(weights, q_params['weights_path'])
                    plt.plot(counter, losses, label='huber loss')
                    plt.plot(counter, scores, label='score')
                    plt.xlabel('Episode')
                    plt.legend()
                    plt.show()
                    pygame.quit()
                    quit()

            game = SpiderJumpGame()
            build_start(game, game.rng)
            init_sequence(game, main_network, 4)
            episode_loss = []
            while (not game.game_over) and episode_frame < 3000:
                # update screen
                set_background(game.displaysurface, game.background)
                game.spider.update(game.plats, game.play_plats)
                # observe frame
                state = main_network.input
                # select action based on epsilon or network
                # epsilon (random exploration) decreases as agent trains for longer
                main_network.epsilon = 1 - (episode * q_params['epsilon_decay_linear'])
                curr_action = [0, 0, 0, 0]
                if random.uniform(0, 1) < main_network.epsilon:
                    action_ind = randint(0, 3)
                else:
                    # print(state.shape)
                    action_ind = main_network.get_highest_q_action(state)
                    # print("action index: ", action_ind)
                curr_action[action_ind] = 1
                # do action
                do_action(game, curr_action)
                if game.spider.rect.top > params.HEIGHT:
                    game.game_over = True
                    game.spider.kill()

                if game.spider.rect.top <= params.HEIGHT / 3:
                    game.spider.pos.y += abs(game.spider.vel.y)
                    for pl in game.plats:
                        pl.rect.y += abs(game.spider.vel.y)
                        if pl.rect.top > params.HEIGHT:
                            pl.kill()

                platforms.plat_gen(game.plats, game.all_sprites, game.play_plats, game.rng)

                game_score = game.font_type.render(str(game.spider.score), True, (123, 255, 0))
                game.displaysurface.blit(game_score, (params.WIDTH / 2, 10))

                for entity in game.all_sprites:
                    entity.draw(game.displaysurface)
                    entity.move()

                pygame.display.update()
                game.FramePerSec.tick(params.FPS)

                # observe reward and next frame
                update_sequence(game, main_network)
                # new_state = main_network.input
                game_frame = to_grayscale(game)
                reward = main_network.get_reward(game.spider, game.game_over)
                terminal = game.game_over
                if reward > 0:
                    episode_frame = 0
                # update memory frames
                replay_memory.add_memory(frame=game_frame, action=action_ind, reward=reward, is_terminal=terminal)
                frame += 1
                episode_frame += 1
                episode_reward += reward
                loss = 0
                # if time to learn
                if frame % q_params['update_frequency'] == 0 and frame > q_params['replay_start']:
                    # print('replayyyy')
                    # replay memory
                    loss = replay(replay_memory, main_network, target_network)
                    if torch.is_tensor(loss):
                        loss = loss.item()
                episode_loss.append(loss)

                # if time to update target network
                if frame % q_params['net_update_frequency'] == 0 and frame > q_params['replay_start']:
                    # update target network
                    target_network.load_state_dict(main_network.state_dict())

            episode += 1
            print(f'Frame {frame}')
            print(f'Game {episode}        Score: {game.spider.score}')
            print('Episode Reward: ' + str(episode_reward))
            episode_reward = 0
            episode_frame = 0
            scores.append(game.spider.score)
            counter.append(episode)
            losses.append(np.mean(episode_loss))
            episode_loss = []
        break

    if q_params['train']:
        weights = main_network.state_dict()
        torch.save(weights, q_params['weights_path'])

    plt.plot(counter, losses)
    plt.xlabel('Episode')
    plt.ylabel('Huber Loss')
    plt.show()

    plt.plot(counter, scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()


if __name__ == '__main__':
    training()
