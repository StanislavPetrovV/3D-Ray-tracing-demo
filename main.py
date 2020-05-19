import pygame
import os
import sys
import tkinter as tk
from ray_tracing import *
from time import time
from settings import *
import control

if __name__ == '__main__':
    os.environ['SDL_VIDEO_WINDOW_POS'] = f'{(tk.Tk().winfo_screenwidth() - WIDTH) // 2},' \
                                         f'{(tk.Tk().winfo_screenheight() - HEIGHT) // 4}'
    pygame.init()
    pygame.display.set_caption('Ray tracing')
    sc = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)
    control = control.Control()

    start = time()
    game = True
    while game:
        start = time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game = False
        control.pressed_keys()
        sc.fill(BLACK)

        for color, xy in ray_tracing(control.key):
            pygame.draw.circle(sc, color, xy, SCALE)

        pygame.display.flip()
        clock.tick()
        print(time() - start)
    pygame.quit()
    sys.exit()
