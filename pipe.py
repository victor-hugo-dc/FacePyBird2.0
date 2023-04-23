import cv2
import numpy as np
from constants import *

class Pipe:
    def __init__(self, ysize: int, inverted: bool = False, xpos: int = None) -> None:
        self.pipe = cv2.imread('assets/sprites/pipe-green.png', cv2.IMREAD_UNCHANGED)
        self.pipe = cv2.resize(self.pipe, (PIPE_WIDTH, PIPE_HEIGHT))

        self.x = xpos or SCREEN_WIDTH

        if inverted:
            self.pipe = cv2.flip(self.pipe, 0)
            self.y = - (self.pipe.shape[0] - ysize)
        
        else:
            self.y = SCREEN_HEIGHT - ysize
        
        self.width = PIPE_WIDTH
        
    def update(self):
        self.x -= GAME_SPEED