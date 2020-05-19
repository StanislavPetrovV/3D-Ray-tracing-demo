import pygame
import numpy as np


class Control:

    def __init__(self):

        self.key = np.array([-1.5, 0.0, -0.5, 0.0, 0.0, 1.0])
        '''key[0] - W S 
           key[1] - A D
           key[2] - W S 
           key[3] - LEFT RIGHT 
           key[4] - UP DOWN
           key[5] - Z X 
        '''

    def pressed_keys(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_w]:
            self.key[0] -= 0.06
        if keys[pygame.K_s]:
            self.key[0] += 0.06
        if keys[pygame.K_a]:
            self.key[1] -= 0.04
        if keys[pygame.K_d]:
            self.key[1] += 0.04
        if keys[pygame.K_q]:
            self.key[2] += 0.04
        if keys[pygame.K_e]:
            self.key[2] -= 0.04
        if keys[pygame.K_LEFT]:
            self.key[3] += 0.02
        if keys[pygame.K_RIGHT]:
            self.key[3] -= 0.02
        if keys[pygame.K_UP]:
            self.key[4] += 0.02
        if keys[pygame.K_DOWN]:
            self.key[4] -= 0.02
        if keys[pygame.K_z]:
            self.key[5] -= 1.0
            self.key[5] = max(1, self.key[5])
        if keys[pygame.K_x]:
            self.key[5] += 1.0
            self.key[5] = min(6.0, self.key[5])