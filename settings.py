import math

WIDTH = 1280
HEIGHT = 720
SCALE = 1
REAL_WIDTH = WIDTH // SCALE
REAL_HEIGHT = HEIGHT // SCALE
HALF_REAL_WIDTH = REAL_WIDTH // 2
HALF_REAL_HEIGHT = REAL_HEIGHT // 2

ASPECT_RATIO = REAL_WIDTH / REAL_HEIGHT #* 0.9
FOV = math.radians(90)
TAN_A = math.tan(FOV / 2)
CAM_DIST = -HALF_REAL_HEIGHT / TAN_A

# camera
PIXEL_SIZE_X, PIXEL_SIZE_Y = (1 / REAL_WIDTH, 1 / REAL_HEIGHT)
PLANE_HALF_WIDTH = TAN_A
PLANE_HALF_HEIGHT = PLANE_HALF_WIDTH / ASPECT_RATIO


# colors
DARKGRAY = (80, 80, 80)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 235)
BLUE_SKY = (0, 189, 255)
BROWN = (140, 50, 20)
DARKBLUE = (0, 0, 70)
DGRAY = (60, 60, 60)
DARKRED = (80, 0, 0)
DARKGREEN = (6, 51, 9)
DARKYELLOW = (155, 135, 12)
GRAY = (100, 100, 100)
GREEN = (0, 225, 0)
LIGHTBLUE = (100, 100, 225)
LIGHTGRAY = (150, 150, 150)
LIGHTGREEN = (100, 255, 100)
RED = (225, 0, 0)
YELLOW = (215, 215, 0)
ORANGE = (243, 107, 8)
PURPLE = (90, 41, 98)