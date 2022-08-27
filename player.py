import pygame
from pygame.locals import *
import params

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # PLAYER SHAPE
        self.surf = pygame.Surface((params.PLAYER_WIDTH, params.PLAYER_HEIGHT))
        self.surf.fill((128, 255, 40))
        self.rect = self.surf.get_rect(center = (10, 420))
        # PLAYER POSITION
        self.pos = params.vec((10, 385))
        self.vel = params.vec(0, 0)
        self.acc = params.vec(0, 0)

    def get_pos(self):
        return self.pos

    def move(self):
        self.acc = params.vec(0, 0)

        pressed_keys = pygame.key.get_pressed()

        if pressed_keys[K_a]:
            self.acc.x = -params.ACC
        if pressed_keys[K_d]:
            self.acc.x = params.ACC

        self.acc.x += self.vel.x * params.FRIC
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc

        if self.pos.x > params.WIDTH:
            self.pos.x = 0
        if self.pos.x < 0:
            self.pos.x = params.WIDTH

        self.rect.midbottom = self.pos

    def draw(self, surface):
        surface.blit(self.surf, self.rect)
