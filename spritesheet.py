import pygame

class SpriteSheet():

    def __init__(self, image):
        self.sheet = image


    def get_image(self, frame, row, width, height, scale):
        image = pygame.Surface((width - 20, height)).convert_alpha()
        image.blit(self.sheet, (0, 0), ((frame * width) + 10, (row * height), width - 10, height))
        image = pygame.transform.scale(image, ((width - 20) * scale, height * scale))
        return image