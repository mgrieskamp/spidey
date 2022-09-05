import pygame
from pygame.locals import *
import params
import spritesheet

vec = pygame.math.Vector2

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # Player appearance
        self.spritesheet_image = pygame.image.load('spider_spritesheet.png').convert_alpha()
        self.spritesheet = spritesheet.SpriteSheet(self.spritesheet_image)
        self.surf = self.spritesheet.get_image(0, 0, 32, 32, 3)
        self.rect = self.surf.get_rect()
        # Player position
        self.pos = vec((10, 385))
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)
        # Score state
        self.score = 0
        # General animation
        self.cooldown = 100
        # Idle left animation
        self.idle_last_update = pygame.time.get_ticks()
        self.idle_l = False
        self.idle_l_list = []
        self.idle_steps = 5
        self.idle_frame = 0
        # Idle right animation
        self.idle_r = True
        self.idle_r_list = []
        # Walk left animation
        self.walk_last_update = pygame.time.get_ticks()
        self.walk_l = False
        self.walk_l_list = []
        self.walk_steps = 6
        self.walk_frame = 0
        # Walk right animation
        self.walk_r = False
        self.walk_r_list = []

        for x in range(self.idle_steps):
            self.idle_l_list.append(self.spritesheet.get_image(x, 0, 32, 32, 3))
            self.idle_r_list.append(self.spritesheet.get_image(x, 8, 32, 32, 3))

        for x in range(self.walk_steps):
            self.walk_l_list.append(self.spritesheet.get_image(x, 1, 32, 32, 3))
            self.walk_r_list.append(self.spritesheet.get_image(x, 9, 32, 32, 3))

        # Jump animation
        self.jumping = False

    def get_movement_coords(self):
        return self.pos, self.vel, self.acc

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

        if -0.2 < self.vel.x < 0.2:
            self.walk_r = False
            self.walk_l = False
        elif self.vel.x < 0:
            self.idle_l = True
            self.idle_r = False
            self.walk_l = True
            self.walk_r = False
        elif self.vel.x > 0:
            self.idle_r = True
            self.idle_l = False
            self.walk_r = True
            self.walk_l = False

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
        if self.walk_l:
            self.walk_last_update, self.walk_frame = \
                self.update_animation(self.walk_l_list, self.walk_frame, self.walk_last_update)
        elif self.walk_r:
            self.walk_last_update, self.walk_frame = \
                self.update_animation(self.walk_r_list, self.walk_frame, self.walk_last_update)
        elif self.idle_l:
            self.idle_last_update, self.idle_frame = \
                self.update_animation(self.idle_l_list, self.idle_frame, self.idle_last_update)
        elif self.idle_r:
            self.idle_last_update, self.idle_frame = \
                self.update_animation(self.idle_r_list, self.idle_frame, self.idle_last_update)

        hits = pygame.sprite.spritecollide(self, sprite_group, False)
        if self.vel.y > 0:
            if hits:
                if self.pos.y < hits[0].rect.bottom:
                    if hits[0].point == True:  # suspicious reference
                        hits[0].point = False
                        self.score += 1
                    self.pos.y = hits[0].rect.top + 1
                    self.vel.y = 0
                    self.jumping = False

    def draw(self, surface):
        if self.walk_l:
            surface.blit(self.walk_l_list[self.walk_frame], self.rect)
        elif self.walk_r:
            surface.blit(self.walk_r_list[self.walk_frame], self.rect)
        elif self.idle_l:
            surface.blit(self.idle_l_list[self.idle_frame], self.rect)
        elif self.idle_r:
            surface.blit(self.idle_r_list[self.idle_frame], self.rect)

    def update_animation(self, animation_list, frame, last_update):
        current_time = pygame.time.get_ticks()
        if current_time - last_update >= self.cooldown:
            frame += 1
            last_update = current_time
            if frame >= len(animation_list):
                frame = 0
        return last_update, frame
