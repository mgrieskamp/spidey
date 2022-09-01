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
        self.speed = random.randint(-1, 1)
        self.moving = True
        self.point = True

    def move(self):
        if self.moving:
            self.rect.move_ip(self.speed, 0)
            if self.speed > 0 and self.rect.left > params.WIDTH:
                self.rect.right = 0
            if self.speed < 0 and self.rect.right < 0:
                self.rect.left = params.WIDTH


def plat_gen(plats, all_sprites):
    while len(plats) < 7:
        width = random.randrange(50, 100)
        pl = Platform()
        close = True

        while close: # fixed freeze ? (height < -50)
            pl = Platform()
            pl.rect.center = (random.randrange(0, params.WIDTH - width), random.randrange(-100, 0))
            close = check(pl, plats)

        plats.add(pl)
        all_sprites.add(pl)


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
