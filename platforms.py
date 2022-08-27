import pygame
from pygame.locals import *
import params

class platform(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.surf = pygame.Surface((params.WIDTH, 20))
        self.surf.fill((255, 0, 0))
        self.rect = self.surf.get_rect(center=(params.WIDTH/2, params.HEIGHT - 10))
