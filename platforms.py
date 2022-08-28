import random
import pygame
from pygame.locals import *
import params


class Platform(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.surf = pygame.Surface((random.randint(50, 100), 12))
        self.surf.fill((0, 255, 0))
        self.rect = self.surf.get_rect(
            center=(random.randint(0, params.WIDTH - 10), random.randint(0, params.HEIGHT - 30)))


def plat_gen(plats, all):
    while len(plats) < 7:
        width = random.randrange(50, 100)
        pl = Platform()
        pl.rect.center = (random.randrange(0, params.WIDTH-width), random.randrange(-50, 0))
        plats.add(pl)
        all.add(plats)