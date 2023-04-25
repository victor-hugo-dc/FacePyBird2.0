import cv2
import numpy as np
from constants import *

class Ground:
    def __init__(self, xpos: int) -> None:
        self.base = cv2.imread('assets/sprites/base.png', -1)
        self.base = cv2.cvtColor(self.base, cv2.COLOR_RGB2RGBA)
        self.base = cv2.resize(self.base, (GROUND_WIDTH, GROUND_HEIGHT))

        self.x = xpos
        self.y = SCREEN_HEIGHT - GROUND_HEIGHT

        self.width = GROUND_WIDTH
        self.game_speed = GAME_SPEED
    
    def update(self):
        self.x -= self.game_speed