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
        close = True

        while close:
            pl = Platform()
            pl.rect.center = (random.randrange(0, params.WIDTH-width), random.randrange(-50, 0))
            close = check(pl, plats)

        plats.add(pl)
        all.add(pl)


def check(plat, all_plats):
    # check for touching platforms
    if pygame.sprite.spritecollideany(plat, all_plats):
        return True
    else:
        # check for too close platforms
        for pl in all_plats:
            if pl == plat:
                continue
            if (abs(plat.rect.top - pl.rect.bottom) < 50) and (abs(plat.rect.bottom - pl.rect.top) < 50):
                return True
        return False
