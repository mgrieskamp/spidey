import random
import pygame
import numpy as np
from pygame.locals import *
import params


class Platform(pygame.sprite.Sprite):
    def __init__(self, rng):
        super().__init__()
        self.image = pygame.image.load('wood_platform.png')
        self.image.set_colorkey((0,0,0))
        self.width = rng.integers(65, 101)
        self.surf = pygame.transform.scale(self.image, (self.width, 12))
        self.rect = self.surf.get_rect(
            center=(rng.integers(51, params.WIDTH - 50), rng.integers(7, params.HEIGHT - 6)))
        self.speed = rng.integers(-1, 2)
        self.moving = False
        self.point = True

    def move(self):
        if self.moving:
            self.rect.move_ip(self.speed, 0)
            if self.speed > 0 and self.rect.left > params.WIDTH:
                self.rect.right = 0
            if self.speed < 0 and self.rect.right < 0:
                self.rect.left = params.WIDTH

    def get_pos(self):
        return self.rect.topleft, self.rect.midtop, self.rect.topright

    def draw(self, surface):
        surface.blit(self.surf, self.rect)


def plat_gen(plats, all_sprites, play_plats, rng):
    while len(plats) < 8:
        width = rng.integers(65, 100)
        pl = Platform(rng)
        close = True

        while close: # fixed freeze ? (height not < -50)
            pl = Platform(rng)
            pl.rect = pl.surf.get_rect(center=(rng.integers(width, params.WIDTH - width), rng.integers(-155, -7)))
            close = check(pl, plats)

        plats.add(pl)
        play_plats.add(pl)
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
