import random
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
platform1 = pygame.sprite.Sprite()
platform1.surf = pygame.Surface((params.WIDTH, 20))
platform1.surf.fill((255, 0, 0))
platform1.rect = platform1.surf.get_rect(center=(params.WIDTH / 2, params.HEIGHT - 10))

all_sprites = pygame.sprite.Group()
all_sprites.add(spider)
all_sprites.add(platform1)

plats = pygame.sprite.Group()
plats.add(platform1)

for x in range(random.randint(5, 6)):
    pl = platforms.Platform()
    close = True
    while close:
        pl = platforms.Platform()
        close = platforms.check(pl, plats)
    plats.add(pl)
    all_sprites.add(pl)

# CODE

running = True
while running:
    spider.update(plats)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Exiting the while loop
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                spider.jump(plats)
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                spider.release_jump()

    if spider.rect.top <= params.HEIGHT / 3:
        spider.pos.y += abs(spider.vel.y)
        for pl in plats:
            pl.rect.y += abs(spider.vel.y)
            if pl.rect.top >= params.HEIGHT:
                pl.kill()

    platforms.plat_gen(plats, all_sprites)
    displaysurface.fill(params.WHITE)

    spider.move()
    for entity in all_sprites:
        displaysurface.blit(entity.surf, entity.rect)

    pygame.display.update()
    FramePerSec.tick(params.FPS)

pygame.quit()
sys.exit()
