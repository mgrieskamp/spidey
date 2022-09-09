import random
import sys
import pygame
from pygame.locals import *
import player
import platforms
import params
import spritesheet
import time
import itertools

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
play_plats = pygame.sprite.Group()
plats.add(platform1)

# Initialize starting screen random platforms
for x in range(random.randint(5, 6)):
    pl = platforms.Platform()
    close = True
    while close:
        pl = platforms.Platform()
        close = platforms.check(pl, plats)
    plats.add(pl)
    play_plats.add(pl)
    all_sprites.add(pl)

# Begin game loop
running = True
while running:
    set_background()
    spider.update(plats, play_plats)

    # Track player inputs
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Exit the while loop
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                spider.jump(plats)
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                spider.release_jump(plats)

    # Initiate game over once the player falls off the screen
    if spider.rect.top > params.HEIGHT:
        for sprite in all_sprites:
            sprite.kill()
            time.sleep(1)
            displaysurface.fill(params.WHITE)
            pygame.display.update()
            time.sleep(1)
            pygame.quit()
            sys.exit()

    # Screen moving (in y-axis) and kills bottom platforms
    if spider.rect.top <= params.HEIGHT / 3:
        spider.pos.y += abs(spider.vel.y)
        for pl in plats:
            pl.rect.y += abs(spider.vel.y)
            if pl.rect.top > params.HEIGHT:
                pl.kill()

    # Generate new random platforms as player moves up
    platforms.plat_gen(plats, all_sprites, play_plats)

    # Set game font and display game score
    game_font_type = pygame.font.SysFont("Verdana", 20)
    game_score = game_font_type.render(str(spider.score), True, (123, 255, 0))
    displaysurface.blit(game_score, (params.WIDTH / 2, 10))

    # Loops through all sprites on screen
    for entity in all_sprites:
        entity.draw(displaysurface)
        entity.move()

    pygame.display.update()
    FramePerSec.tick(params.FPS)

pygame.quit()
sys.exit()
