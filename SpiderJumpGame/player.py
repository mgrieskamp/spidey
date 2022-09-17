import pygame
from pygame.locals import *
import params
import spritesheet
import os

vec = pygame.math.Vector2


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # Player appearance
        self.spritesheet_image = pygame.image.load(os.path.join('SpiderJumpGame', 'spider_spritesheet.png')).convert_alpha()
        self.spritesheet = spritesheet.SpriteSheet(self.spritesheet_image)
        self.surf = self.spritesheet.get_image(0, 0, 32, 32, 3)
        self.rect = self.surf.get_rect()
        # Player position
        self.pos = vec((190, 435))
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

        self.jumping_l_list = []
        self.jumping_r_list = []
        self.jumping_frame = 0
        self.jumping_last_update = pygame.time.get_ticks()
        self.jumping_l = False
        self.jumping_r = False
        self.end_jump = False
        self.mid_jump = False
        self.start_jump = False

        for x in range(0, 9):
            self.jumping_l_list.append(self.spritesheet.get_image(x, 2, 32, 32, 3))
            self.jumping_r_list.append(self.spritesheet.get_image(x, 10, 32, 32, 3))

        # Check if points should be given for landing on a new platform
        self.new_landing = False
        # Check if spider is on platform
        self.on_platform = False

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

    def agent_move(self, action):
        self.acc = vec(0, 0.5)

        if action[0] == 1:
            self.acc.x = -params.ACC
        if action[1] == 1:
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
            self.start_jump = True
            self.jumping_frame = 0

    def release_jump(self, sprite_group):
        if self.jumping:
            if self.vel.y < -3:
                self.vel.y = -3
            self.jumping_frame = 3
            self.start_jump = False
            self.mid_jump = True

    def update(self, plats, play_plats):
        if self.start_jump:
            if self.idle_l:
                self.jumping_last_update, self.jumping_frame = \
                    self.update_animation(self.jumping_l_list, self.jumping_frame, self.jumping_last_update, 3, True)
            else:
                self.jumping_last_update, self.jumping_frame = \
                    self.update_animation(self.jumping_r_list, self.jumping_frame, self.jumping_last_update, 3, True)
        elif self.mid_jump:
            if self.idle_l:
                self.jumping_last_update, self.jumping_frame = \
                    self.update_animation(self.jumping_l_list, self.jumping_frame, self.jumping_last_update, 6, True)
            else:
                self.jumping_last_update, self.jumping_frame = \
                    self.update_animation(self.jumping_r_list, self.jumping_frame, self.jumping_last_update, 6, True)
        elif self.walk_l:
            self.walk_last_update, self.walk_frame = \
                self.update_animation(self.walk_l_list, self.walk_frame, self.walk_last_update, len(self.walk_l_list))
        elif self.walk_r:
            self.walk_last_update, self.walk_frame = \
                self.update_animation(self.walk_r_list, self.walk_frame, self.walk_last_update, len(self.walk_r_list))
        elif self.idle_l:
            self.idle_last_update, self.idle_frame = \
                self.update_animation(self.idle_l_list, self.idle_frame, self.idle_last_update, len(self.idle_l_list))
        elif self.idle_r:
            self.idle_last_update, self.idle_frame = \
                self.update_animation(self.idle_r_list, self.idle_frame, self.idle_last_update, len(self.idle_r_list))

        self.new_landing = False
        self.on_platform = False
        hits = pygame.sprite.spritecollide(self, play_plats, False)
        if hits:
            if self.pos.y < hits[0].rect.bottom:
                self.on_platform = True
        hits = pygame.sprite.spritecollide(self, plats, False)
        if self.vel.y > 0:
            if hits:
                if self.pos.y < hits[0].rect.bottom:
                    if hits[0].point == True:  # suspicious reference
                        hits[0].point = False
                        hits[0].image = pygame.image.load(os.path.join('SpiderJumpGame', 'visited_platform.png'))
                        hits[0].surf = pygame.transform.scale(hits[0].image, (hits[0].width, 12))
                        self.score += 1
                        self.new_landing = True
                    self.pos.y = hits[0].rect.top + 1
                    self.vel.y = 0
                    self.jumping = False
                    self.mid_jump = False
                    self.jumping_frame = 6
                    if self.jumping_l:
                        self.jumping_last_update, self.jumping_frame = \
                            self.update_animation(self.jumping_l_list, self.jumping_frame, self.jumping_last_update, 9,
                                                  True)
                    else:
                        self.jumping_last_update, self.jumping_frame = \
                            self.update_animation(self.jumping_r_list, self.jumping_frame, self.jumping_last_update, 9,
                                                  True)
                    self.jumping_l = False
                    self.jumping_r = False

    def draw(self, surface):
        if self.jumping_l:
            surface.blit(self.jumping_l_list[self.jumping_frame], self.rect)
        elif self.jumping_r:
            surface.blit(self.jumping_r_list[self.jumping_frame], self.rect)
        elif self.walk_l:
            surface.blit(self.walk_l_list[self.walk_frame], self.rect)
        elif self.walk_r:
            surface.blit(self.walk_r_list[self.walk_frame], self.rect)
        elif self.idle_l:
            surface.blit(self.idle_l_list[self.idle_frame], self.rect)
        elif self.idle_r:
            surface.blit(self.idle_r_list[self.idle_frame], self.rect)

    def update_animation(self, animation_list, frame, last_update, frame_limit, stop=False):
        current_time = pygame.time.get_ticks()
        if current_time - last_update >= self.cooldown:
            frame += 1
            last_update = current_time
            if frame >= frame_limit:
                if stop:
                    return last_update, frame
                frame = 0
        return last_update, frame
