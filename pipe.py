import cv2
import numpy as np
from constants import *

class Pipe:
    def __init__(self, ysize: int, inverted: bool = False, xpos: int = None) -> None:
        self.pipe: np.ndarray = cv2.imread('assets/sprites/pipe-green.png', cv2.IMREAD_UNCHANGED)
        self.pipe: np.ndarray = cv2.resize(self.pipe, (PIPE_WIDTH, PIPE_HEIGHT))

        self.x: int = xpos or SCREEN_WIDTH

        if inverted:
            self.pipe = cv2.flip(self.pipe, 0)
            self.y = - (self.pipe.shape[0] - ysize)
        
        else:
            self.y = SCREEN_HEIGHT - ysize
        
        self.height: int = PIPE_HEIGHT
        self.width: int = PIPE_WIDTH

        self.scored: bool = False

        self.game_speed = GAME_SPEED
        
    def update(self):
        self.x -= self.game_speed