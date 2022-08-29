import pygame
from pygame.locals import *
import params

vec = pygame.math.Vector2

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # PLAYER SHAPE
        self.surf = pygame.Surface((params.PLAYER_WIDTH, params.PLAYER_HEIGHT))
        self.surf.fill((128, 255, 40))
        self.rect = self.surf.get_rect()
        # PLAYER POSITION
        self.pos = vec((10, 385))
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)
        # CHECK JUMP STATE
        self.jumping = False

    def get_pos(self):
        return self.pos.x, self.pos.y

    def move(self):
        self.acc = vec(0, 0.5)

        pressed_keys = pygame.key.get_pressed()

        if pressed_keys[K_a]:
            self.acc.x = -params.ACC
        if pressed_keys[K_d]:
            self.acc.x = params.ACC

        old_posx = self.pos.x

        self.acc.x += self.vel.x * params.FRIC
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc

        if self.pos.x > params.WIDTH:
            self.pos.x = old_posx
        if self.pos.x < 0:
            self.pos.x = old_posx

        self.rect.midbottom = self.pos

    def jump(self, sprite_group):
        hits = pygame.sprite.spritecollide(self, sprite_group, False)
        if hits and not self.jumping:
            self.vel.y = -15
            self.jumping = True

    def release_jump(self):
        if self.jumping:
            if self.vel.y < -3:
                self.vel.y = -3

    def update(self, sprite_group):
        hits = pygame.sprite.spritecollide(self, sprite_group, False)
        if self.vel.y > 0:
            if hits:
                if self.pos.y < hits[0].rect.bottom:
                    self.pos.y = hits[0].rect.top + 1
                    self.vel.y = 0
                    self.jumping = False

    def draw(self, surface):
        surface.blit(self.surf, self.rect)