import sys
import pygame
from pygame.locals import *
import player
import platforms
import params

pygame.init()
vec = pygame.math.Vector2  # 2 for two dimensional

FramePerSec = pygame.time.Clock()

displaysurface = pygame.display.set_mode((params.WIDTH, params.HEIGHT))
displaysurface.fill(params.WHITE)
pygame.display.set_caption("Game")

spider = player.Player()
platform1 = platforms.platform()

all_sprites = pygame.sprite.Group()
all_sprites.add(spider)
all_sprites.add(platform1)

platforms = pygame.sprite.Group()
platforms.add(platform1)

# CODE

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Exiting the while loop
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                spider.jump(platforms)

    displaysurface.fill(params.WHITE)

    spider.move()
    spider.update(platforms)
    for entity in all_sprites:
        displaysurface.blit(entity.surf, entity.rect)

    pygame.display.update()
    FramePerSec.tick(params.FPS)

pygame.quit()
sys.exit()
